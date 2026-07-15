from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .quant import QuantizedTensor, dequantize_int8_per_block, estimate_quantized_bytes, quantize_int8_per_block


@dataclass(frozen=True)
class PagedKVCache:
    """分页存储的 INT8 KV cache。

    这个结构模拟 LLM decode 阶段常见的 paged KV cache：
    每条序列的 KV 不要求在物理内存中连续，而是按固定大小的 block 存储。
    `block_table` 负责把“逻辑 block 编号”映射到实际存放 KV 的“物理 block 编号”。

    字段说明：
    - k_quant/v_quant: 量化后的物理 K/V block。量化前的物理布局是
      [physical_blocks, heads, block_size, head_dim]。
    - block_table: [batch, logical_blocks]，其中
      block_table[b, logical_block] 表示 batch b 的某个逻辑 block
      实际存放在哪个 physical block；值为 -1 表示尚未分配。
    - lengths: [batch]，每条序列当前真实有效 token 数。
    - block_size: 每个逻辑/物理 block 容纳的 token 数。
    - max_seq_len: 当前 cache 中需要还原或参与 attention 的最大序列长度。
    """

    k_quant: QuantizedTensor
    v_quant: QuantizedTensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    block_size: int
    max_seq_len: int

    @property
    def batch_size(self) -> int:
        """cache 中的 batch 数量。"""

        return int(self.block_table.shape[0])

    @property
    def num_heads(self) -> int:
        """每个 token 的 attention head 数量。"""

        return int(self.k_quant.original_shape[1])

    @property
    def head_dim(self) -> int:
        """每个 attention head 的特征维度。"""

        return int(self.k_quant.original_shape[-1])

    @classmethod
    def from_dense(
        cls,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int = 16,
        *,
        lengths: torch.Tensor | None = None,
        block_table_pattern: str = "contiguous",
        layout_seed: int = 0,
    ) -> "PagedKVCache":
        """从 dense KV tensor 构建分页 INT8 KV cache。

        参数：
        - k/v: dense KV，形状为 [batch, heads, seq_len, head_dim]。
        - block_size: 每个分页 block 包含的 token 数。
        - lengths: 可选的每条 request 有效长度；用于 profiling variable length decode。
        - block_table_pattern: 物理 block 布局。``contiguous`` 保持默认连续布局，
          ``shuffled`` 随机打乱物理 block，``interleaved`` 按 logical block 交错排列。
        - layout_seed: ``shuffled`` 布局的随机种子。

        返回：
        - PagedKVCache，内部会把 dense KV 按 sequence 维度切成 block，
          重排为物理 block 布局后再做 per-block INT8 量化。

        默认实现使用最简单的一一映射：
        batch 内每条序列的 logical block 都对应一个独立 physical block，
        不做 block 复用或复杂内存分配。profiling 布局只改变 physical block
        的内存排列和 block_table 映射，不改变逻辑 KV 内容。
        """

        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("k 和 v 必须是 [batch, heads, seq_len, head_dim] 形状")
        if k.shape != v.shape:
            raise ValueError("k 和 v 的形状必须一致")
        if block_size <= 0:
            raise ValueError("block_size 必须为正数")

        # batch: 序列数量；heads: attention head 数；
        # seq_len: 每条序列的 token 数；head_dim: 单个 head 的维度。
        batch, heads, seq_len, head_dim = k.shape

        # 每条序列需要多少个 logical block；不足一个 block 的尾部会补零。
        blocks_per_seq = (seq_len + block_size - 1) // block_size

        # 当前简单实现中，每个 batch 的每个 logical block 都分配一个 physical block。
        total_blocks = batch * blocks_per_seq

        # padding 后的 sequence 长度，保证能被 block_size 整除。
        padded_seq_len = blocks_per_seq * block_size
        if padded_seq_len != seq_len:
            # 只在 sequence 维度补零；这些 padding token 会通过 lengths 在 attention 中屏蔽。
            pad_shape = (batch, heads, padded_seq_len - seq_len, head_dim)
            k = torch.cat([k, k.new_zeros(pad_shape)], dim=2)
            v = torch.cat([v, v.new_zeros(pad_shape)], dim=2)

        # dense K 的原始布局是 [batch, heads, seq, dim]。
        # 先 reshape 出 logical block 维度，再把布局变成
        # [batch, logical_block, heads, block_offset, dim]，
        # 最后把 batch 和 logical_block 合并成 physical block 维度：
        # [physical_blocks, heads, block_size, head_dim]。
        physical_k = (
            k.reshape(batch, heads, blocks_per_seq, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(total_blocks, heads, block_size, head_dim)
        )

        # V 使用和 K 完全相同的物理 block 布局。
        physical_v = (
            v.reshape(batch, heads, blocks_per_seq, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(total_blocks, heads, block_size, head_dim)
        )

        # block_table[b, logical_block] = physical_block。
        # 基线 physical block 按 batch-major 顺序连续编号。
        block_table = torch.arange(total_blocks, dtype=torch.int64, device=k.device).reshape(batch, blocks_per_seq)

        pattern = block_table_pattern.strip().lower()
        if pattern in {"contiguous", ""}:
            pass
        elif pattern in {"shuffled", "random"}:
            generator = torch.Generator(device=k.device)
            generator.manual_seed(int(layout_seed))
            perm = torch.randperm(total_blocks, generator=generator, device=k.device)
            physical_k = physical_k[perm].contiguous()
            physical_v = physical_v[perm].contiguous()
            inverse = torch.empty_like(perm)
            inverse[perm] = torch.arange(total_blocks, dtype=torch.int64, device=k.device)
            block_table = inverse.reshape(batch, blocks_per_seq)
        elif pattern == "interleaved":
            # 让同一 request 的 logical block 在 physical store 中跨 batch 交错，
            # 用于测 block_table 间接寻址和 L2 locality 对 Kernel 2 的影响。
            perm = torch.arange(total_blocks, dtype=torch.int64, device=k.device).reshape(batch, blocks_per_seq)
            perm = perm.transpose(0, 1).contiguous().reshape(-1)
            physical_k = physical_k[perm].contiguous()
            physical_v = physical_v[perm].contiguous()
            inverse = torch.empty_like(perm)
            inverse[perm] = torch.arange(total_blocks, dtype=torch.int64, device=k.device)
            block_table = inverse.reshape(batch, blocks_per_seq)
        else:
            raise ValueError("block_table_pattern 必须是 contiguous、shuffled 或 interleaved")

        if lengths is None:
            effective_lengths = torch.full((batch,), seq_len, dtype=torch.int64, device=k.device)
        else:
            effective_lengths = lengths.to(device=k.device, dtype=torch.int64).contiguous()
            if effective_lengths.ndim != 1 or effective_lengths.shape[0] != batch:
                raise ValueError("lengths 必须是 [batch] 形状")
            if bool((effective_lengths < 0).any().item()) or bool((effective_lengths > seq_len).any().item()):
                raise ValueError("lengths 中的有效长度必须位于 [0, seq_len]")

        return cls(
            # 对物理 K/V block 分别做 per-block INT8 量化。
            k_quant=quantize_int8_per_block(physical_k, block_size=block_size),
            v_quant=quantize_int8_per_block(physical_v, block_size=block_size),
            block_table=block_table,

            # 默认每条序列长度相同；profiling 可传入 variable lengths。
            lengths=effective_lengths,
            block_size=block_size,
            max_seq_len=seq_len,
        )

    def to_dense(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """把分页 INT8 KV cache 还原为 dense K/V。

        返回：
        - dense_k/dense_v: 形状均为 [batch, heads, max_seq_len, head_dim]。

        还原流程：
        1. 先把量化的 physical K/V block 反量化为 float32。
        2. 根据 block_table 为每条序列收集对应的 physical block。
        3. 把布局从 [batch, logical_blocks, heads, block_size, head_dim]
           转回 [batch, heads, seq, head_dim]。
        4. 截断到 max_seq_len，去掉末尾 padding block 中不需要的位置。
        """

        # physical_k/v: [physical_blocks, heads, block_size, head_dim]。
        physical_k = dequantize_int8_per_block(self.k_quant, dtype=torch.float32)
        physical_v = dequantize_int8_per_block(self.v_quant, dtype=torch.float32)

        # block_table 中 -1 表示未分配 block。这里 clamp 到 0 是为了让 gather 有合法索引；
        # 后续 attention 会通过 lengths 屏蔽无效 token，因此未分配位置不会影响有效输出。
        table = self.block_table.clamp_min(0)

        # gathered_k/v: [batch, logical_blocks, heads, block_size, head_dim]。
        gathered_k = physical_k[table]
        gathered_v = physical_v[table]

        # 转回 dense attention 常用布局 [batch, heads, seq, head_dim]，
        # 并裁剪掉由于 block 对齐产生的多余 padding token。
        dense_k = (
            gathered_k.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(self.batch_size, self.num_heads, -1, self.head_dim)
            [..., : self.max_seq_len, :]
        )
        dense_v = (
            gathered_v.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(self.batch_size, self.num_heads, -1, self.head_dim)
            [..., : self.max_seq_len, :]
        )
        return dense_k, dense_v

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "PagedKVCache":
        """向分页 cache 追加新的 dense KV token。

        参数：
        - k_new/v_new: 新增 token 的 dense KV，形状为
          [batch, heads, tokens, head_dim]。

        返回：
        - 新的 PagedKVCache。由于 dataclass 是 frozen=True，这里不会原地修改
          当前 cache，而是构造并返回更新后的 cache。

        当前可移植实现会：
        1. 反量化已有 physical block。
        2. 按每条序列当前 length 找到追加 token 应写入的 logical block 和 offset。
        3. 必要时扩展 block_table 并分配新的 physical block。
        4. 写入新增 KV 后重新量化整个 physical block store。

        它没有从完整 dense [batch, heads, seq, dim] 重新构建 cache，
        但为了保证 CPU/CUDA 行为确定，仍然会反量化并重新量化物理 block。
        """

        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError("k_new 和 v_new 必须是 [batch, heads, tokens, head_dim] 形状")
        if k_new.shape != v_new.shape:
            raise ValueError("k_new 和 v_new 的形状必须一致")
        if k_new.shape[0] != self.batch_size or k_new.shape[1] != self.num_heads or k_new.shape[3] != self.head_dim:
            raise ValueError("新增 KV 的 batch、heads 和 head_dim 必须与 cache 对齐")

        # append_tokens: 本次要给每条序列追加的 token 数。
        append_tokens = int(k_new.shape[2])
        if append_tokens <= 0:
            raise ValueError("追加 token 维度不能为空")
        if k_new.device != self.block_table.device or v_new.device != self.block_table.device:
            raise ValueError("新增 KV tensor 必须与 cache 位于同一设备")

        # 反量化已有 physical block，并 clone 出可写副本。
        # physical_k/v 的形状是 [physical_blocks, heads, block_size, head_dim]。
        physical_k = dequantize_int8_per_block(self.k_quant, dtype=torch.float32).clone()
        physical_v = dequantize_int8_per_block(self.v_quant, dtype=torch.float32).clone()

        # table/lengths 都 clone，避免修改当前 frozen cache 内部引用的 tensor。
        table = self.block_table.clone()
        lengths = self.lengths.clone()

        # 追加后每条序列需要的 logical block 数。
        # 如果任意序列需要更多 block_table 列，就统一扩展 table。
        required_blocks = ((lengths + append_tokens + self.block_size - 1) // self.block_size).max()
        required_cols = int(required_blocks.item())
        if required_cols > table.shape[1]:
            # 新增的 logical block 先标记为 -1，表示尚未分配 physical block。
            pad = table.new_full((self.batch_size, required_cols - table.shape[1]), -1)
            table = torch.cat([table, pad], dim=1)

        # 追加写入时使用 float32，之后再整体量化回 INT8 block store。
        k_new = k_new.to(dtype=torch.float32)
        v_new = v_new.to(dtype=torch.float32)
        for batch_idx in range(self.batch_size):
            # start 是该 batch 序列追加前的末尾位置。
            start = int(lengths[batch_idx].item())
            for token_idx in range(append_tokens):
                # position 是新 token 在该序列中的绝对 token 位置。
                position = start + token_idx

                # logical_block: position 所在的逻辑 block 编号。
                # block_offset: token 在该 block 内部的偏移。
                logical_block = position // self.block_size
                block_offset = position % self.block_size

                # 查询 logical block 对应的 physical block。
                physical_idx = int(table[batch_idx, logical_block].item())
                if physical_idx < 0:
                    # 如果该 logical block 还没有物理存储，就在末尾分配一个新的 physical block。
                    physical_idx = int(physical_k.shape[0])
                    empty_shape = (1, self.num_heads, self.block_size, self.head_dim)
                    physical_k = torch.cat([physical_k, physical_k.new_zeros(empty_shape)], dim=0)
                    physical_v = torch.cat([physical_v, physical_v.new_zeros(empty_shape)], dim=0)
                    table[batch_idx, logical_block] = physical_idx

                # 将新 token 写入对应 physical block 的 block_offset 位置。
                physical_k[physical_idx, :, block_offset, :] = k_new[batch_idx, :, token_idx, :]
                physical_v[physical_idx, :, block_offset, :] = v_new[batch_idx, :, token_idx, :]

        # 所有 batch 的序列长度统一增加 append_tokens。
        lengths = lengths + append_tokens
        return PagedKVCache(
            # 写入后的 physical block store 重新量化为 INT8。
            k_quant=quantize_int8_per_block(physical_k, block_size=self.block_size),
            v_quant=quantize_int8_per_block(physical_v, block_size=self.block_size),
            block_table=table,
            lengths=lengths,
            block_size=self.block_size,

            # attention 还原 dense KV 时最多需要看到新的最大有效长度。
            max_seq_len=int(lengths.max().item()),
        )

    def estimated_bytes(self) -> int:
        """估算当前 paged quant KV cache 占用的字节数。

        统计范围包括：
        - 量化后的 K block store。
        - 量化后的 V block store。
        - block_table 的索引存储。
        - lengths 的序列长度存储。

        这是用于 benchmark/report 的近似值，不等价于 PyTorch tensor
        实际 allocator 占用的全部显存或内存。
        """

        # block_table 保存 logical block 到 physical block 的 int64 映射。
        table_bytes = self.block_table.numel() * self.block_table.element_size()

        # lengths 保存每条序列当前有效 token 数。
        length_bytes = self.lengths.numel() * self.lengths.element_size()
        return estimate_quantized_bytes(self.k_quant) + estimate_quantized_bytes(self.v_quant) + table_bytes + length_bytes
