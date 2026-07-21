"""Triton fused INT8 KV attention.

本文件实现 Kernel 1：K/V cache 是连续 dense sequence 布局，但已经按 block
量化为 INT8。Triton kernel 不先把 K/V 反量化成完整 dense tensor，而是在每个
[batch, head] 的 program 内边读 INT8、边反量化、边做 decode attention。

整体数据流：
1. Python launcher 接收 q 和 QuantizedTensor 形式的 k_quant/v_quant。
2. kernel 读取当前 [batch, head] 的 q 向量。
3. 按 token tile 扫描历史 K，load INT8 values 和每个量化 block 的 scale/zero_point。
4. 在寄存器中反量化 K，计算 q @ k，并用 online softmax 维护 m/l/acc。
5. 用同一套地址读取并反量化 V，累积 softmax(QK) @ V。
6. 长序列可走 Split-K：多个 split kernel 并行产出 partial_m/l/acc，再由 combine 合并。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import os

import torch

from .quant import QuantizedTensor, estimate_quantized_bytes
from .triton_utils import HAS_TRITON, next_power_of_2, tl, triton


@dataclass(frozen=True)
class FusedKVBuffers:
    """两个 fused kernel 共享的 query / INT8 KV / lengths 输入指针。

    把 8 个 tensor 参数收拢成一个对象，避免 launcher 里逐个平铺、错位传参。
    字段顺序与 kernel signature 的前 8 个位置参数严格对应。
    """

    q: torch.Tensor
    k_values: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_values: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    lengths: torch.Tensor

    def as_args(self) -> tuple:
        """按 kernel signature 顺序返回输入 tensor 的位置参数元组。"""

        return (
            self.q,
            self.k_values,
            self.k_scale,
            self.k_zero,
            self.v_values,
            self.v_scale,
            self.v_zero,
            self.lengths,
        )


@dataclass(frozen=True)
class FusedAttentionMeta:
    """两个 fused kernel 共享的 constexpr 元参数与 launch 配置。

    这些值决定 kernel 的地址计算、softmax 缩放、tile 尺寸和 warp 数。
    core_args() 给出单 kernel 也需要的公共 constexpr；split kernel 额外的
    num_splits/chunk_tokens 由 launcher 单独插入，避免污染公共顺序。
    """

    heads: int
    seq_len: int
    head_dim: int
    n_blocks: int
    block_size: int
    sm_scale: float
    has_lengths: bool
    num_splits: int
    chunk_tokens: int
    block_n: int
    block_d: int
    num_warps: int

    def core_args(self) -> tuple:
        """返回单 kernel 与 split kernel 公共的 constexpr（不含 Split-K 专有项）。"""

        return (
            self.heads,
            self.seq_len,
            self.head_dim,
            self.n_blocks,
            self.block_size,
            self.sm_scale,
            self.has_lengths,
        )


if HAS_TRITON:

    @triton.jit
    def _pid_to_batch_head(pid, heads: tl.constexpr):
        """把展平后的 program id 还原成 batch/head 下标。"""

        batch_idx = pid // heads
        head_idx = pid - batch_idx * heads
        return batch_idx, head_idx

    @triton.jit
    def _make_dim_offsets(head_dim: tl.constexpr, block_d: tl.constexpr):
        """生成 head_dim 维度上的向量化偏移和 padding mask。"""

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        return offs_d, d_mask

    @triton.jit
    def _load_query(
        q_ptr,
        batch_idx,
        head_idx,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        offs_d,
        d_mask,
    ):
        """读取当前 [batch, head] 的 query 向量，并返回它的展平 base 地址。"""

        q_base = (batch_idx * heads + head_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        return q_base, q

    @triton.jit
    def _effective_length(
        lengths_ptr,
        batch_idx,
        seq_len: tl.constexpr,
        has_lengths: tl.constexpr,
    ):
        """返回当前 batch 的有效 sequence 长度。"""

        effective_len = seq_len
        if has_lengths:
            loaded_len = tl.load(lengths_ptr + batch_idx)
            effective_len = tl.minimum(loaded_len, seq_len)
        return effective_len

    @triton.jit
    def _fused_attention_tile(
        q,
        m,
        l,
        acc,
        k_values_ptr,
        k_scale_ptr,
        k_zero_point_ptr,
        v_values_ptr,
        v_scale_ptr,
        v_zero_point_ptr,
        batch_idx,
        head_idx,
        offs_s,
        s_mask,
        offs_d,
        d_mask,
        heads: tl.constexpr,
        n_blocks: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        sm_scale: tl.constexpr,
    ):
        """处理一个 token tile，并更新 online softmax 状态。

        该 helper 同时服务单 kernel 和 Split-K split kernel。调用方负责提供
        offs_s/s_mask，因此单 kernel 可以传完整长度 mask，split kernel 可以额外
        叠加 split_end mask。函数内部只关注连续 INT8 KV 的地址计算、反量化和
        attention 状态更新。
        """

        # 将 sequence 位置映射到量化 block 编号和 block 内偏移。
        # quant_block: token 属于第几个量化 block。
        # block_offset: token 在这个量化 block 内的 offset。
        quant_block = offs_s // block_size
        block_offset = offs_s - quant_block * block_size

        # QuantizedTensor.values 的布局是：
        # [batch, heads, n_blocks, block_size, head_dim]。
        # kv_offsets 是二维地址矩阵 [block_n, block_d]，指向当前 token tile 的 K/V 元素。
        kv_offsets = (
            ((((batch_idx * heads + head_idx) * n_blocks + quant_block[:, None]) * block_size + block_offset[:, None])
             * head_dim)
            + offs_d[None, :]
        )
        # kv_mask 同时屏蔽无效 token 和 head_dim padding 列。
        kv_mask = s_mask[:, None] & d_mask[None, :]

        # scale/zero_point 的布局是 [batch, heads, n_blocks, 1, 1]，
        # 因此展平后每个 [batch, head, quant_block] 对应一个参数。
        # qparam_offsets 是 [block_n]，每个 token 对应一个量化参数地址。
        qparam_offsets = (batch_idx * heads + head_idx) * n_blocks + quant_block
        k_scale = tl.load(k_scale_ptr + qparam_offsets, mask=s_mask, other=1.0).to(tl.float32)
        k_zero = tl.load(k_zero_point_ptr + qparam_offsets, mask=s_mask, other=0).to(tl.float32)

        # values 以 signed int8 保存，真实 uint8 逻辑值需要 +128。
        # 反量化公式：x = (uint8_value - zero_point) * scale。
        # k_deq 形状 [block_n, block_d]，只在寄存器中存在。
        k_i8 = tl.load(k_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
        k_deq = (k_i8 + 128.0 - k_zero[:, None]) * k_scale[:, None]

        # scores: [block_n]，表示当前 q 对这一段历史 K 的注意力分数。
        scores = tl.sum(k_deq * q[None, :], axis=1) * sm_scale
        scores = tl.where(s_mask, scores, -3.4028234663852886e38)

        # online softmax 更新。这样不需要先保存完整 [seq_len] scores。
        block_m = tl.max(scores, axis=0)
        new_m = tl.maximum(m, block_m)
        old_scale = tl.exp(m - new_m)
        probs = tl.exp(scores - new_m)
        probs = tl.where(s_mask, probs, 0.0)
        new_l = l * old_scale + tl.sum(probs, axis=0)

        # V 使用和 K 相同的 int8/scale/zero_point 布局与反量化公式。
        v_scale = tl.load(v_scale_ptr + qparam_offsets, mask=s_mask, other=1.0).to(tl.float32)
        v_zero = tl.load(v_zero_point_ptr + qparam_offsets, mask=s_mask, other=0).to(tl.float32)
        v_i8 = tl.load(v_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
        # v_deq 形状 [block_n, block_d]；与 probs[:, None] 相乘后按 token 维求和。
        v_deq = (v_i8 + 128.0 - v_zero[:, None]) * v_scale[:, None]

        # 累积 PV。acc 始终和当前 new_m 对齐，避免 softmax 溢出。
        acc = acc * old_scale + tl.sum(probs[:, None] * v_deq, axis=0)
        return new_m, new_l, acc

    @triton.jit
    def _normalize_acc(acc, l):
        """将未归一化 PV 累积除以 softmax 分母；空序列输出 0。"""

        denom = tl.where(l > 0.0, l, 1.0)
        out = acc / denom
        return tl.where(l > 0.0, out, 0.0)

    @triton.jit
    def _fused_dequant_attention_kernel(
        q_ptr,
        k_values_ptr,
        k_scale_ptr,
        k_zero_point_ptr,
        v_values_ptr,
        v_scale_ptr,
        v_zero_point_ptr,
        lengths_ptr,
        out_ptr,
        heads: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        n_blocks: tl.constexpr,
        block_size: tl.constexpr,
        sm_scale: tl.constexpr,
        has_lengths: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        """Triton Kernel 1：直接读取 INT8 K/V，融合反量化和 decode attention。

        函数作用：
        - 这是 fused 路径的单 kernel 快路径。
        - grid 只有一维，大小为 batch * heads。
        - 每个 program 计算一个 batch 中一个 attention head 的输出向量。

        参数含义：
        - q_ptr: 输入 query，形状 [batch, heads, head_dim]。
        - k_values_ptr/v_values_ptr: 量化后的 INT8 K/V values，逻辑形状
          [batch, heads, n_blocks, block_size, head_dim]。
        - k_scale_ptr/v_scale_ptr: K/V 每个量化 block 的 scale，逻辑形状
          [batch, heads, n_blocks, 1, 1]，展平后按 [batch, head, block] 访问。
        - k_zero_point_ptr/v_zero_point_ptr: K/V 每个量化 block 的 zero_point，
          与 scale 同布局，zero_point 位于 uint8 域。
        - lengths_ptr: 可选有效长度 [batch]；has_lengths=False 时不会读取它。
        - out_ptr: 输出 tensor，形状 [batch, heads, head_dim]。
        - heads: attention head 数，用于从 pid 还原 batch/head。
        - seq_len: K/V cache 的最大历史 token 数。
        - head_dim: 每个 head 的真实维度。
        - n_blocks: 每条 sequence 被量化成的 block 数。
        - block_size: 量化 block 在 sequence 维度覆盖的 token 数。
        - sm_scale: softmax scale，通常是 1 / sqrt(head_dim)。
        - has_lengths: 是否启用 variable length mask。
        - block_n: 每轮扫描的 token tile 大小。
        - block_d: head_dim 向上取到 2 的幂后的 tile 大小；超出 head_dim 的列用 mask 屏蔽。

        每个 Triton program 负责一个 [batch, head] 的 decode attention。
        kernel 内部按 sequence block 分块扫描历史 K/V：
        1. 从 int8 K block 读取数据。
        2. 用对应 scale/zero_point 在寄存器中反量化。
        3. 与当前 q 做 QK 点积。
        4. 用 online softmax 累积概率归一化项和 PV 结果。
        5. 再读取 int8 V block、反量化，并累积最终输出。

        关键点：这里不会构造完整 dense FP16/FP32 K/V tensor。
        """

        # pid 对应展平后的 [batch, head] 编号。
        pid = tl.program_id(0)

        # 从 pid 还原 batch 下标和 head 下标：
        # pid = batch_idx * heads + head_idx。
        batch_idx, head_idx = _pid_to_batch_head(pid, heads)

        # 当前 head 内的 feature/head_dim 下标。
        # offs_d 长度是 block_d，可能大于真实 head_dim；d_mask 屏蔽 padding lane。
        offs_d, d_mask = _make_dim_offsets(head_dim, block_d)

        # q 的布局是 [batch, heads, head_dim]，这里读取当前 [batch, head] 的 q 向量。
        # q_base 是这个向量在展平内存中的起始元素下标。
        q_base, q = _load_query(q_ptr, batch_idx, head_idx, heads, head_dim, offs_d, d_mask)

        # effective_len 表示当前 batch 真实参与 attention 的 token 数。
        # 如果传入 lengths，则每个 batch 可以有不同有效长度；否则默认 seq_len 全有效。
        effective_len = _effective_length(lengths_ptr, batch_idx, seq_len, has_lengths)

        # online softmax 的状态：
        # m 是当前已扫描 scores 的最大值；
        # l 是 exp(scores - m) 的累积和；
        # acc 是 softmax(scores) @ V 的未归一化累积结果。
        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 每次扫描 block_n 个历史 token。block_n 不需要等于量化 block_size；
        # scale/zero_point 会按 token 所在的量化 block 单独读取。
        for start in range(0, seq_len, block_n):
            # offs_s: 当前 token tile 内的绝对 sequence 下标，形状 [block_n]。
            # s_mask: token 是否处于当前 batch 的有效长度范围内。
            offs_s = start + tl.arange(0, block_n)
            s_mask = offs_s < effective_len

            m, l, acc = _fused_attention_tile(
                q,
                m,
                l,
                acc,
                k_values_ptr,
                k_scale_ptr,
                k_zero_point_ptr,
                v_values_ptr,
                v_scale_ptr,
                v_zero_point_ptr,
                batch_idx,
                head_idx,
                offs_s,
                s_mask,
                offs_d,
                d_mask,
                heads,
                n_blocks,
                block_size,
                head_dim,
                sm_scale,
            )

        # 如果 effective_len 为 0，l 会保持 0；这种异常输入下输出置 0。
        # 先保护分母，避免先执行 acc / 0 再 where 造成无效浮点中间值。
        out = _normalize_acc(acc, l)

        # 输出布局是 [batch, heads, head_dim]，与 q 一致。
        tl.store(out_ptr + q_base + offs_d, out, mask=d_mask)

    @triton.jit
    def _fused_dequant_attention_split_kernel(
        q_ptr,
        k_values_ptr,
        k_scale_ptr,
        k_zero_point_ptr,
        v_values_ptr,
        v_scale_ptr,
        v_zero_point_ptr,
        lengths_ptr,
        partial_m_ptr,
        partial_l_ptr,
        partial_acc_ptr,
        heads: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        n_blocks: tl.constexpr,
        block_size: tl.constexpr,
        sm_scale: tl.constexpr,
        has_lengths: tl.constexpr,
        num_splits: tl.constexpr,
        chunk_tokens: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        """Split-K split kernel：grid=(batch*heads, num_splits)。

        函数作用：
        - 这是 fused 路径的长序列并行版本的第一阶段。
        - grid 第 0 维是 batch*heads，第 1 维是 split_id。
        - 每个 program 只处理同一个 [batch, head] 的一段 token 区间。
        - 输出不是最终 attention，而是该 token 段的 online softmax 状态。

        参数含义：
        - q_ptr/k_values_ptr/k_scale_ptr/k_zero_point_ptr/v_values_ptr/v_scale_ptr/
          v_zero_point_ptr/lengths_ptr: 含义与 _fused_dequant_attention_kernel 相同。
        - partial_m_ptr: [batch*heads, num_splits]，每段 scores 的局部最大值。
        - partial_l_ptr: [batch*heads, num_splits]，每段 exp(scores - partial_m) 的和。
        - partial_acc_ptr: [batch*heads, num_splits, head_dim]，每段未归一化 PV 累积。
        - num_splits: sequence 被拆成多少段并行计算。
        - chunk_tokens: 每段覆盖的 token 数，launcher 会向上对齐到 block_n 的倍数。

        数据逻辑：
        - split kernel 和单 kernel 使用同一套地址计算、反量化和 online softmax。
        - 区别是这里只扫描 [split_start, split_end)。
        - combine kernel 后续把各段 partial_m/l/acc 按 softmax 数学重新归一化。

        与单 kernel 的唯一区别：每个 program 只扫自己那段 token
        `[split_id*chunk_tokens, (split_id+1)*chunk_tokens)`，产出该段的
        online-softmax 部分状态 (partial_m, partial_l, partial_acc) 写入 scratch，
        不直接写 out。合并交给 _combine_splits_kernel。

        chunk_tokens 由 launcher 保证是 block_n 的整数倍，因此内层循环覆盖的范围
        恰好等于本段 token 区间，尾部用 effective_len mask 处理不整除。
        """

        # pid: 展平后的 [batch, head] 编号。
        # split_id: 当前 program 负责的 sequence 分段编号。
        pid = tl.program_id(0)
        split_id = tl.program_id(1)

        batch_idx, head_idx = _pid_to_batch_head(pid, heads)

        offs_d, d_mask = _make_dim_offsets(head_dim, block_d)

        # q_base 指向当前 [batch, head] query 向量。
        q_base, q = _load_query(q_ptr, batch_idx, head_idx, heads, head_dim, offs_d, d_mask)

        # effective_len 用于 variable length；超过该长度的 token 不参与当前 split。
        effective_len = _effective_length(lengths_ptr, batch_idx, seq_len, has_lengths)

        # 本段负责的 token 区间。split_end 不超过 seq_len。
        # 例如 seq_len=2048,num_splits=4 时，通常每段约 512 个 token。
        split_start = split_id * chunk_tokens
        split_end = tl.minimum(split_start + chunk_tokens, seq_len)

        # 当前 split 内部的 online softmax 局部状态。
        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 只扫本段。chunk_tokens 是 block_n 的整数倍，循环次数编译期已知。
        for local in range(0, chunk_tokens, block_n):
            # local 是 split 内的相对 offset；offs_s 转成全局 sequence 下标。
            offs_s = split_start + local + tl.arange(0, block_n)
            s_mask = (offs_s < effective_len) & (offs_s < split_end)

            m, l, acc = _fused_attention_tile(
                q,
                m,
                l,
                acc,
                k_values_ptr,
                k_scale_ptr,
                k_zero_point_ptr,
                v_values_ptr,
                v_scale_ptr,
                v_zero_point_ptr,
                batch_idx,
                head_idx,
                offs_s,
                s_mask,
                offs_d,
                d_mask,
                heads,
                n_blocks,
                block_size,
                head_dim,
                sm_scale,
            )

        # 写出本段 partial 状态到 scratch，供 combine kernel 合并。
        # partial_m/l 布局 [batch*heads, num_splits]，partial_acc 布局 [..., head_dim]。
        out_idx = pid * num_splits + split_id
        tl.store(partial_m_ptr + out_idx, m)
        tl.store(partial_l_ptr + out_idx, l)
        tl.store(partial_acc_ptr + out_idx * head_dim + offs_d, acc, mask=d_mask)

    @triton.jit
    def _combine_splits_kernel(
        partial_m_ptr,
        partial_l_ptr,
        partial_acc_ptr,
        out_ptr,
        num_splits: tl.constexpr,
        head_dim: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        """Split-K combine：把某个 [batch, head] 的 S 段 partial 状态合并成最终输出。

        函数作用：
        - 这是 fused Split-K 的第二阶段。
        - 每个 program 读取同一个 [batch, head] 的所有 split partial。
        - 通过跨 split 的 online softmax rescale 得到与单 kernel 数学等价的输出。

        参数含义：
        - partial_m_ptr: split kernel 写出的局部最大值 [batch*heads, num_splits]。
        - partial_l_ptr: split kernel 写出的局部分母 [batch*heads, num_splits]。
        - partial_acc_ptr: split kernel 写出的局部 PV 累积 [batch*heads, num_splits, head_dim]。
        - out_ptr: 最终输出 [batch, heads, head_dim] 的展平指针。
        - num_splits: 要合并的 split 数。
        - head_dim/block_d: 真实 head 维度和向上取 2 的幂后的计算 tile。

        每个 program 负责一个 [batch, head]，读取自己 S 段的 (partial_m, partial_l,
        partial_acc)，做跨段 online-softmax rescale。数学与 kernel 内跨 block 的
        合并完全同构，只是这里跨的是 split 段。
        """

        # pid 仍然是展平后的 [batch, head] 编号；out_ptr 也按这个顺序写回。
        pid = tl.program_id(0)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        # 合并过程中的全局 online softmax 状态。
        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 逐段合并。空段的 partial_m=-FLT_MAX、partial_l=0，rescale 后贡献为 0，
        # 不会污染结果；全空时 l 保持 0，最后走除零保护输出 0。
        for s in range(0, num_splits):
            # idx 是当前 [batch, head, split] 在 partial_m/l 中的展平下标。
            idx = pid * num_splits + s
            pm = tl.load(partial_m_ptr + idx)
            pl = tl.load(partial_l_ptr + idx)
            pacc = tl.load(partial_acc_ptr + idx * head_dim + offs_d, mask=d_mask, other=0.0)

            # 将旧全局状态和当前 split 状态 rescale 到共同最大值 new_m 后相加。
            new_m = tl.maximum(m, pm)
            old_scale = tl.exp(m - new_m)
            cur_scale = tl.exp(pm - new_m)
            l = l * old_scale + pl * cur_scale
            acc = acc * old_scale + pacc * cur_scale
            m = new_m

        out = _normalize_acc(acc, l)
        tl.store(out_ptr + pid * head_dim + offs_d, out, mask=d_mask)


def _validate_triton_fused_inputs(
    q: torch.Tensor,
    k_quant: QuantizedTensor,
    v_quant: QuantizedTensor,
    lengths: torch.Tensor | None,
) -> None:
    """校验 Triton fused attention 的输入约束。

    函数作用：
    - 在 Python 侧提前检查 shape/device/长度约束。
    - 避免把不匹配的 tensor 传入 JIT kernel 后产生难定位的 CUDA 访问错误。

    参数含义：
    - q: decode 阶段的 query，形状 [batch, heads, head_dim]。
    - k_quant/v_quant: 由 quantize_int8_per_block 生成的量化 K/V。
      original_shape 必须是 [batch, heads, seq_len, head_dim]。
    - lengths: 可选 [batch] 有效长度；用于 variable length attention mask。

    Triton Kernel 1 当前实现聚焦 decode attention 主路径：
    - q 必须是 [batch, heads, head_dim]；
    - 量化 K/V 必须来自 [batch, heads, seq_len, head_dim]；
    - 所有输入必须在同一个 CUDA 设备；
    - head_dim 当前限制为不超过 256，覆盖 TODO.svg 中要求的 64/128。
    """

    if q.ndim != 3:
        raise ValueError("q 必须是 [batch, heads, head_dim] 形状")
    if len(k_quant.original_shape) != 4 or len(v_quant.original_shape) != 4:
        raise ValueError("k_quant 和 v_quant 必须来自 [batch, heads, seq_len, head_dim] 张量")
    if k_quant.original_shape != v_quant.original_shape:
        raise ValueError("k_quant 和 v_quant 的 original_shape 必须一致")
    if tuple(q.shape) != (
        k_quant.original_shape[0],
        k_quant.original_shape[1],
        k_quant.original_shape[3],
    ):
        raise ValueError("q 的 batch、heads 和 head_dim 必须与量化 K/V 对齐")
    if q.device.type != "cuda":
        raise RuntimeError("Triton fused attention 需要 CUDA q tensor")
    if k_quant.device != q.device or v_quant.device != q.device:
        raise RuntimeError("q、k_quant 和 v_quant 必须位于同一个 CUDA 设备")
    if lengths is not None:
        if lengths.ndim != 1 or lengths.shape[0] != q.shape[0]:
            raise ValueError("lengths 必须是 [batch] 形状")
        if lengths.device != q.device:
            raise RuntimeError("lengths 必须与 q 位于同一个 CUDA 设备")
    if q.shape[-1] > 256:
        raise ValueError("Triton fused attention 当前只支持 head_dim <= 256")


# Split-K 默认参数；实验结论见 doc/optimization-log.md。
# 矩阵 profiling 显示 s2048/s4096 的 fused 最优点都在 block_n=128,
# num_warps=4, split=4。短序列仍保留单 kernel 快路径，避免 combine 开销。
_DEFAULT_BLOCK_N = 128
_DEFAULT_NUM_SPLITS = 4
_TOKENS_PER_SPLIT = 512
_S_MAX = 32


def _compute_num_splits(seq_len: int, block_n: int) -> int:
    """返回 Split-K 段数，默认长序列使用 4 段。

    函数作用：
    - 根据 seq_len 和 block_n 选择是否启用 Split-K。
    - 返回值 S 决定 launcher 走单 kernel 还是 split+combine 两阶段。

    参数含义：
    - seq_len: K/V cache 的最大 sequence 长度。
    - block_n: kernel 每轮扫描的 token tile 大小。

    返回：
    - int，Split-K 段数。1 表示不开 Split-K，>1 表示并行拆段。

    S==1 时退化回单 kernel 快路径（零 combine 开销）。短序列(<=512)
    仍用 1 段；A100 full matrix 显示 s2048/s4096 的稳定最优候选是
    split=4，因此未设置环境变量时长序列默认 4 段。

    A/B 开关：环境变量 FLASHSPEC_NUM_SPLITS 可覆盖自适应值，用于在
    num_warps 固定的前提下隔离 Split-K 的纯贡献：
    - FLASHSPEC_NUM_SPLITS=1  强制关闭 Split-K（走单 kernel 快路径）；
    - FLASHSPEC_NUM_SPLITS=8  强制 8 段；
    - 未设置或非法值           走自适应逻辑。
    S 会被 clamp 到 [1, seq_len 能切出的最大段数]，避免空段过多。
    """

    override = os.environ.get("FLASHSPEC_NUM_SPLITS")
    if override is not None:
        try:
            forced = int(override)
        except ValueError:
            forced = 0
        if forced >= 1:
            # 每段至少覆盖 block_n 个 token，否则切出的段是纯空段。
            max_splits = max(1, (seq_len + block_n - 1) // block_n)
            return max(1, min(forced, max_splits))

    max_splits = max(1, (seq_len + block_n - 1) // block_n)
    if seq_len <= _TOKENS_PER_SPLIT:
        return 1
    s = min(_DEFAULT_NUM_SPLITS, max_splits)
    return max(1, min(s, _S_MAX))


def _resolve_block_n() -> int:
    """每个 program 每轮扫描的 token 数。默认 128。

    profiling matrix 结论：block_n=128 的有效 DRAM throughput 最好；
    block_n=32 虽然 occupancy 更高但明显更慢。A/B 开关仍保留：
    寄存器占用与 block_n 成正比。调小 block_n 可降低 registers_per_thread、
    抬高占用率天花板，代价是循环轮数和 MIO/scoreboard 压力增加。
    - FLASHSPEC_BLOCK_N=32  每轮扫 32 token（更少寄存器）；
    - FLASHSPEC_BLOCK_N=64  旧默认；
    - FLASHSPEC_BLOCK_N=128 当前默认；
    - 未设置或非法值        用默认 128。
    限制为 [16, 128] 内的 2 的幂，避免病态 tile 尺寸。
    """

    override = os.environ.get("FLASHSPEC_BLOCK_N")
    if override is not None:
        try:
            v = int(override)
        except ValueError:
            v = 0
        if v in (16, 32, 64, 128):
            return v
    return _DEFAULT_BLOCK_N


def _resolve_num_warps() -> int:
    """Triton program 的 warp 数。默认 4，可用环境变量做 profiling sweep。

    早期 A100 实测显示 8 warp 比 4 warp 慢，但后续会结合 block_n/Split-K
    做矩阵验证；因此这里保留开关，避免每次实验都改代码。
    """

    override = os.environ.get("FLASHSPEC_NUM_WARPS")
    if override is not None:
        try:
            v = int(override)
        except ValueError:
            v = 0
        if v in (1, 2, 4, 8):
            return v
    return 4


def _run_fused_dequant_attention_triton(
    q: torch.Tensor,
    k_quant: QuantizedTensor,
    v_quant: QuantizedTensor,
    lengths: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """运行真正的 Triton Kernel 1，不 materialize dense K/V。

    函数作用：
    - 这是 Python 层的 fused Triton launcher。
    - 负责输入校验、contiguous 视图整理、元参数计算、scratch 分配和 kernel launch。
    - 根据 num_splits 自动选择单 kernel 快路径或 Split-K 两阶段路径。

    参数含义：
    - q: [batch, heads, head_dim] 的 query。
    - k_quant/v_quant: 连续 KV cache 的 INT8 量化表示。
    - lengths: 可选 [batch] 有效长度；None 表示每条序列都使用完整 seq_len。
    - return_stats: 是否返回 benchmark/report 用统计字段。

    返回：
    - return_stats=False: out，形状 [batch, heads, head_dim]。
    - return_stats=True: (out, stats)，stats 包含压缩率、是否物化 dense KV、kernel 参数等。

    数据逻辑：
    - q/k/v 只做 contiguous，不创建 dense FP K/V。
    - 单 kernel 路径：每个 [batch, head] 一个 program，直接写 out。
    - Split-K 路径：第一阶段写 partial_m/l/acc，第二阶段 combine 后写 out。

    这个函数负责 Python 侧的参数整理、输出 tensor 分配和 kernel launch。
    kernel 内部直接读取 QuantizedTensor.values/scale/zero_point，并在寄存器中完成
    INT8 反量化、QK、softmax 和 PV。
    """

    _validate_triton_fused_inputs(q, k_quant, v_quant, lengths)

    # contiguous 只整理现有输入布局，不生成 dense K/V 中间结果。
    # q_contig: [batch, heads, head_dim]。
    q_contig = q.contiguous()
    # k_values/v_values: [batch, heads, n_blocks, block_size, head_dim] 的 INT8 数据。
    k_values = k_quant.values.contiguous()
    # k_scale/v_scale: [batch, heads, n_blocks, 1, 1] 的 FP32 scale。
    k_scale = k_quant.scale.contiguous()
    # k_zero/v_zero: [batch, heads, n_blocks, 1, 1] 的 int16 zero point。
    k_zero = k_quant.zero_point.contiguous()
    v_values = v_quant.values.contiguous()
    v_scale = v_quant.scale.contiguous()
    v_zero = v_quant.zero_point.contiguous()
    # lengths_contig 在 has_lengths=False 时传 q_contig 作为占位指针；kernel 不会读取。
    lengths_contig = lengths.contiguous() if lengths is not None else q_contig

    # batch: request 数；heads: 每个 request 的 attention head 数；head_dim: 单 head 维度。
    batch, heads, head_dim = q_contig.shape
    # seq_len: 量化前 K/V 的真实 sequence 长度；n_blocks: 量化 block 数。
    seq_len = int(k_quant.original_shape[-2])
    n_blocks = int(k_values.shape[-3])
    # block_size: 每个量化 block 覆盖多少个 token。
    block_size = int(k_quant.block_size)
    # block_d: Triton tile 的 feature 维度，取 2 的幂便于 tl.arange 和向量化。
    block_d = next_power_of_2(int(head_dim))

    # block_n 是每个 program 每轮扫描的 token 数（实验 4：可用 FLASHSPEC_BLOCK_N 覆盖）。
    # tile [block_n, block_d] 越小，registers_per_thread 越低、占用率天花板越高。
    block_n = _resolve_block_n()
    num_warps = _resolve_num_warps()
    # out 与 q 同形状同 dtype；kernel 内部 FP32 累积，写回时按 out dtype 转换。
    out = torch.empty_like(q_contig)
    # sm_scale 是 scaled dot-product attention 的 1/sqrt(d) 缩放。
    sm_scale = 1.0 / math.sqrt(float(head_dim))

    # num_splits 控制数据路径：1=单 kernel，>1=Split-K split+combine。
    num_splits = _compute_num_splits(seq_len, block_n)
    # chunk_tokens 只有 Split-K 路径用到；单 kernel 时值不参与 launch。
    # 向上取整到 block_n 的整数倍，使 split kernel 内层循环恰好覆盖本段。
    chunk_tokens = ((seq_len + num_splits - 1) // num_splits + block_n - 1) // block_n * block_n

    # 把逐个平铺的 tensor 指针和 constexpr 元参数收拢成两个对象，
    # 让单 kernel 和 Split-K 两阶段从同一份参数按固定顺序组装 launch。
    buffers = FusedKVBuffers(
        q=q_contig,
        k_values=k_values,
        k_scale=k_scale,
        k_zero=k_zero,
        v_values=v_values,
        v_scale=v_scale,
        v_zero=v_zero,
        lengths=lengths_contig,
    )
    meta = FusedAttentionMeta(
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        n_blocks=n_blocks,
        block_size=block_size,
        sm_scale=sm_scale,
        has_lengths=lengths is not None,
        num_splits=num_splits,
        chunk_tokens=chunk_tokens,
        block_n=block_n,
        block_d=block_d,
        num_warps=num_warps,
    )

    if num_splits == 1:
        # S==1 快路径：直接用原单 kernel 写 out，不分配 scratch、不启 combine。
        grid = (batch * heads,)
        _fused_dequant_attention_kernel[grid](
            *buffers.as_args(),
            out,
            *meta.core_args(),
            meta.block_n,
            meta.block_d,
            num_warps=meta.num_warps,
        )
    else:
        # Split-K：把 seq_len 切成 num_splits 段并行，grid=(batch*heads, S)。
        # chunk_tokens 已在上面按 block_n 对齐，使 split kernel 内层循环恰好
        # 覆盖本段，尾段用 mask 处理不整除。
        # rows 是展平后的 [batch, head] 数，也是 combine kernel 的 program 数。
        rows = batch * heads
        # scratch：partial 状态，不是 dense KV。
        # partial_m: 每个 row 每个 split 的局部最大 score。
        # partial_l: 每个 row 每个 split 的局部 softmax 分母。
        # partial_acc: 每个 row 每个 split 的局部未归一化 PV 累积。
        # s4096/d128,S=8 时 acc 约 16.8MB。
        partial_m = torch.empty((rows, num_splits), device=q_contig.device, dtype=torch.float32)
        partial_l = torch.empty((rows, num_splits), device=q_contig.device, dtype=torch.float32)
        partial_acc = torch.empty((rows, num_splits, head_dim), device=q_contig.device, dtype=torch.float32)

        _fused_dequant_attention_split_kernel[(rows, num_splits)](
            *buffers.as_args(),
            partial_m,
            partial_l,
            partial_acc,
            *meta.core_args(),
            meta.num_splits,
            meta.chunk_tokens,
            meta.block_n,
            meta.block_d,
            num_warps=meta.num_warps,
        )
        _combine_splits_kernel[(rows,)](
            partial_m,
            partial_l,
            partial_acc,
            out,
            meta.num_splits,
            meta.head_dim,
            meta.block_d,
            num_warps=meta.num_warps,
        )

    if not return_stats:
        return out

    dense_bytes = 2 * batch * heads * seq_len * head_dim * q_contig.element_size()
    quant_bytes = estimate_quantized_bytes(k_quant) + estimate_quantized_bytes(v_quant)
    stats = {
        "dense_kv_bytes": float(dense_bytes),
        "quant_kv_bytes": float(quant_bytes),
        "compression_ratio": float(dense_bytes / max(1, quant_bytes)),
        "materializes_dense_kv": 0.0,
        # Split-K 段数：1 表示走单 kernel 快路径，>1 表示 split+combine 路径。
        "num_splits": float(num_splits),
        # 每轮扫描 token 数（实验 4 降寄存器 A/B 变量）。
        "block_n": float(block_n),
        # 每个 Triton program 的 warp 数（实验 5 profiling matrix 变量）。
        "num_warps": float(num_warps),
    }
    return out, stats


def fused_dequant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 1 的 Triton 兼容入口：fused dequant attention。

    函数作用：
    - 对外暴露统一入口，调用方不需要判断当前环境是否有 Triton/CUDA。
    - CUDA + Triton 时走真正 fused kernel；否则走 PyTorch fallback 方便测试。

    参数/返回：
    - 透传给 _run_fused_dequant_attention_triton 或 attention.fused_dequant_attention。
    - 返回形状与 q 相同的 attention 输出，或 return_stats=True 时返回 (out, stats)。

    当前行为：
    - 如果安装了 Triton 且输入位于 CUDA，则启动真正的 fused dequant kernel。
    - 否则延迟导入并调用 PyTorch 参考实现，保证 CPU 环境仍可导入和 smoke test。
    """

    if HAS_TRITON and _args and isinstance(_args[0], torch.Tensor) and _args[0].device.type == "cuda":
        return _run_fused_dequant_attention_triton(*_args, **_kwargs)

    # 非 CUDA 或未安装 Triton 时使用 portable fallback；这个 fallback 会 materialize dense KV。
    from .attention import fused_dequant_attention

    return fused_dequant_attention(*_args, **_kwargs)
