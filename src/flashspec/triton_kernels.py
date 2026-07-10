from __future__ import annotations

from typing import Any

import math

import torch

from .paged import PagedKVCache
from .quant import QuantizedTensor, estimate_quantized_bytes


# Triton 是可选依赖：
# - 安装了 `.[triton]` 时，可以在这里接入真正的 Triton kernel。
# - 未安装 Triton 时，项目仍然应该能在 CPU/普通 PyTorch 环境中导入和测试。
try:
    # triton: Triton Python API，通常用于定义 @triton.jit kernel 和 launch。
    import triton  # type: ignore

    # tl: Triton language namespace，真实 kernel 中会用 tl.load/tl.store/tl.dot 等。
    import triton.language as tl  # type: ignore

    # HAS_TRITON 标记当前环境是否可以使用 Triton 后端。
    HAS_TRITON = True
except ModuleNotFoundError:
    # 没有安装 Triton 时，保留同名变量，避免其他模块引用时报 NameError。
    triton = None
    tl = None

    # 当前环境只能使用 portable PyTorch fallback。
    HAS_TRITON = False


if HAS_TRITON:

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

        # 从 pid 还原 batch 下标和 head 下标。
        batch_idx = pid // heads
        head_idx = pid - batch_idx * heads

        # 当前 head 内的 feature/head_dim 下标。
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        # q 的布局是 [batch, heads, head_dim]，这里读取当前 [batch, head] 的 q 向量。
        q_base = (batch_idx * heads + head_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        # 如果传入 lengths，则每个 batch 可以有不同有效长度；否则默认 seq_len 全有效。
        effective_len = seq_len
        if has_lengths:
            loaded_len = tl.load(lengths_ptr + batch_idx)
            effective_len = tl.minimum(loaded_len, seq_len)

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
            offs_s = start + tl.arange(0, block_n)
            s_mask = offs_s < effective_len

            # 将 sequence 位置映射到量化 block 编号和 block 内偏移。
            quant_block = offs_s // block_size
            block_offset = offs_s - quant_block * block_size

            # QuantizedTensor.values 的布局是：
            # [batch, heads, n_blocks, block_size, head_dim]。
            kv_offsets = (
                ((((batch_idx * heads + head_idx) * n_blocks + quant_block[:, None]) * block_size + block_offset[:, None])
                 * head_dim)
                + offs_d[None, :]
            )
            kv_mask = s_mask[:, None] & d_mask[None, :]

            # scale/zero_point 的布局是 [batch, heads, n_blocks, 1, 1]，
            # 因此展平后每个 [batch, head, quant_block] 对应一个参数。
            qparam_offsets = (batch_idx * heads + head_idx) * n_blocks + quant_block
            k_scale = tl.load(k_scale_ptr + qparam_offsets, mask=s_mask, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_point_ptr + qparam_offsets, mask=s_mask, other=0).to(tl.float32)

            # values 以 signed int8 保存，真实 uint8 逻辑值需要 +128。
            # 反量化公式：x = (uint8_value - zero_point) * scale。
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
            v_deq = (v_i8 + 128.0 - v_zero[:, None]) * v_scale[:, None]

            # 累积 PV。acc 始终和当前 new_m 对齐，避免 softmax 溢出。
            acc = acc * old_scale + tl.sum(probs[:, None] * v_deq, axis=0)
            m = new_m
            l = new_l

        # 如果 effective_len 为 0，l 会保持 0；这种异常输入下输出置 0。
        # 先保护分母，避免先执行 acc / 0 再 where 造成无效浮点中间值。
        denom = tl.where(l > 0.0, l, 1.0)
        out = acc / denom
        out = tl.where(l > 0.0, out, 0.0)

        # 输出布局是 [batch, heads, head_dim]，与 q 一致。
        tl.store(out_ptr + q_base + offs_d, out, mask=d_mask)

    @triton.jit
    def _paged_quant_attention_kernel(
        q_ptr,
        k_values_ptr,
        k_scale_ptr,
        k_zero_point_ptr,
        v_values_ptr,
        v_scale_ptr,
        v_zero_point_ptr,
        block_table_ptr,
        lengths_ptr,
        out_ptr,
        heads: tl.constexpr,
        max_seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        table_cols: tl.constexpr,
        physical_quant_blocks: tl.constexpr,
        page_block_size: tl.constexpr,
        quant_block_size: tl.constexpr,
        sm_scale: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        """Triton Kernel 2：直接按 block_table 读取 paged INT8 KV 并执行 attention。

        每个 Triton program 负责一个 [batch, head] 的 decode attention。
        与 Kernel 1 的区别是：K/V 不再按 dense sequence 顺序连续存储，而是
        被拆成 physical block。kernel 必须先把 token 位置映射成 logical block，
        再通过 block_table 查到 physical block，最后从该 physical block 中读取
        INT8 K/V。

        关键点：
        - 不调用 cache.to_dense()。
        - 不 gather 出完整 dense K/V。
        - 在 kernel 内完成 block_table 间接寻址、INT8 反量化和 attention。
        """

        # pid 对应展平后的 [batch, head] 编号。
        pid = tl.program_id(0)

        # 从 pid 还原 batch 下标和 head 下标。
        batch_idx = pid // heads
        head_idx = pid - batch_idx * heads

        # 当前 head 内的 feature/head_dim 下标。
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        # 读取当前 [batch, head] 的 query 向量，布局为 [batch, heads, head_dim]。
        q_base = (batch_idx * heads + head_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        # 每条 request 可以有不同有效长度。effective_len 决定 attention 可见 token 范围。
        loaded_len = tl.load(lengths_ptr + batch_idx)
        effective_len = tl.minimum(loaded_len, max_seq_len)

        # online softmax 状态，避免保存完整 scores。
        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 按逻辑 sequence 位置扫描；每个 token 再通过 block_table 映射到物理 block。
        for start in range(0, max_seq_len, block_n):
            offs_s = start + tl.arange(0, block_n)
            s_mask = offs_s < effective_len

            # logical_block 是 token 属于该 request 的第几个逻辑页。
            # page_offset 是 token 在该页内的位置。
            logical_block = offs_s // page_block_size
            page_offset = offs_s - logical_block * page_block_size

            # block_table 布局是 [batch, table_cols]。
            # 未分配的 logical block 通常是 -1；有效长度 mask 会避免访问无效 token。
            table_mask = s_mask & (logical_block < table_cols)
            physical_block = tl.load(
                block_table_ptr + batch_idx * table_cols + logical_block,
                mask=table_mask,
                other=0,
            )
            physical_valid = table_mask & (physical_block >= 0)

            # physical block 内部的 sequence 位置还要映射到量化 block。
            # 当前 PagedKVCache 默认 page_block_size == quant_block_size，因此
            # physical_quant_block 通常为 0；这里保留通用写法，方便后续扩展。
            physical_quant_block = page_offset // quant_block_size
            quant_offset = page_offset - physical_quant_block * quant_block_size
            qparam_mask = physical_valid & (physical_quant_block < physical_quant_blocks)

            # QuantizedTensor.values 物理布局：
            # [physical_blocks, heads, physical_quant_blocks, quant_block_size, head_dim]。
            kv_offsets = (
                ((((physical_block[:, None] * heads + head_idx) * physical_quant_blocks + physical_quant_block[:, None])
                  * quant_block_size + quant_offset[:, None])
                 * head_dim)
                + offs_d[None, :]
            )
            kv_mask = qparam_mask[:, None] & d_mask[None, :]

            # scale/zero_point 布局是 [physical_blocks, heads, physical_quant_blocks, 1, 1]。
            qparam_offsets = (physical_block * heads + head_idx) * physical_quant_blocks + physical_quant_block
            k_scale = tl.load(k_scale_ptr + qparam_offsets, mask=qparam_mask, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_point_ptr + qparam_offsets, mask=qparam_mask, other=0).to(tl.float32)

            # INT8 values 以 signed int8 存储，反量化时先 +128 回到 uint8 域。
            k_i8 = tl.load(k_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
            k_deq = (k_i8 + 128.0 - k_zero[:, None]) * k_scale[:, None]

            # QK 点积得到当前 block_n 个 token 的 attention scores。
            scores = tl.sum(k_deq * q[None, :], axis=1) * sm_scale
            scores = tl.where(qparam_mask, scores, -3.4028234663852886e38)

            # online softmax 更新：只保留当前最大值、分母和 PV 累积。
            block_m = tl.max(scores, axis=0)
            new_m = tl.maximum(m, block_m)
            old_scale = tl.exp(m - new_m)
            probs = tl.exp(scores - new_m)
            probs = tl.where(qparam_mask, probs, 0.0)
            new_l = l * old_scale + tl.sum(probs, axis=0)

            # V 走同样的 paged 间接寻址和 INT8 反量化。
            v_scale = tl.load(v_scale_ptr + qparam_offsets, mask=qparam_mask, other=1.0).to(tl.float32)
            v_zero = tl.load(v_zero_point_ptr + qparam_offsets, mask=qparam_mask, other=0).to(tl.float32)
            v_i8 = tl.load(v_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
            v_deq = (v_i8 + 128.0 - v_zero[:, None]) * v_scale[:, None]

            # 累积 PV。acc 始终和当前 new_m 对齐。
            acc = acc * old_scale + tl.sum(probs[:, None] * v_deq, axis=0)
            m = new_m
            l = new_l

        # 空长度 request 输出 0，避免除零。
        denom = tl.where(l > 0.0, l, 1.0)
        out = acc / denom
        out = tl.where(l > 0.0, out, 0.0)

        # 输出布局是 [batch, heads, head_dim]。
        tl.store(out_ptr + q_base + offs_d, out, mask=d_mask)


def _next_power_of_2(x: int) -> int:
    """返回不小于 x 的最小 2 的幂，用作 Triton block_d。"""

    return 1 << (x - 1).bit_length()


def _validate_triton_fused_inputs(
    q: torch.Tensor,
    k_quant: QuantizedTensor,
    v_quant: QuantizedTensor,
    lengths: torch.Tensor | None,
) -> None:
    """校验 Triton fused attention 的输入约束。

    Triton Kernel 1 当前实现聚焦 decode attention 主路径：
    - q 必须是 [batch, heads, head_dim]；
    - 量化 K/V 必须来自 [batch, heads, seq_len, head_dim]；
    - 所有输入必须在同一个 CUDA 设备；
    - head_dim 当前限制为不超过 256，覆盖 TODO 中要求的 64/128。
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


def _run_fused_dequant_attention_triton(
    q: torch.Tensor,
    k_quant: QuantizedTensor,
    v_quant: QuantizedTensor,
    lengths: torch.Tensor | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """运行真正的 Triton Kernel 1，不 materialize dense K/V。

    这个函数负责 Python 侧的参数整理、输出 tensor 分配和 kernel launch。
    kernel 内部直接读取 QuantizedTensor.values/scale/zero_point，并在寄存器中完成
    INT8 反量化、QK、softmax 和 PV。
    """

    _validate_triton_fused_inputs(q, k_quant, v_quant, lengths)

    # contiguous 只整理现有输入布局，不生成 dense K/V 中间结果。
    q_contig = q.contiguous()
    k_values = k_quant.values.contiguous()
    k_scale = k_quant.scale.contiguous()
    k_zero = k_quant.zero_point.contiguous()
    v_values = v_quant.values.contiguous()
    v_scale = v_quant.scale.contiguous()
    v_zero = v_quant.zero_point.contiguous()
    lengths_contig = lengths.contiguous() if lengths is not None else q_contig

    batch, heads, head_dim = q_contig.shape
    seq_len = int(k_quant.original_shape[-2])
    n_blocks = int(k_values.shape[-3])
    block_size = int(k_quant.block_size)
    block_d = _next_power_of_2(int(head_dim))

    # block_n 是每个 program 每轮扫描的 token 数。64 在 head_dim=64/128 下
    # 能控制单个 program 的寄存器/临时矩阵规模，同时覆盖长 seq_len 的循环扫描。
    block_n = 64
    out = torch.empty_like(q_contig)

    grid = (batch * heads,)
    _fused_dequant_attention_kernel[grid](
        q_contig,
        k_values,
        k_scale,
        k_zero,
        v_values,
        v_scale,
        v_zero,
        lengths_contig,
        out,
        heads,
        seq_len,
        head_dim,
        n_blocks,
        block_size,
        1.0 / math.sqrt(float(head_dim)),
        lengths is not None,
        block_n,
        block_d,
        num_warps=4,
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
    }
    return out, stats


def _validate_triton_paged_inputs(q: torch.Tensor, cache: PagedKVCache) -> None:
    """校验 Triton paged attention 的输入约束。

    Kernel 2 当前实现聚焦 paged decode attention 主路径：
    - q 必须是 [batch, heads, head_dim]；
    - cache 内的 K/V 必须是 PagedKVCache 的物理 block 布局；
    - q、量化 K/V、block_table、lengths 必须位于同一个 CUDA 设备；
    - head_dim 当前限制为不超过 256，覆盖常见 64/128 配置。
    """

    if q.ndim != 3:
        raise ValueError("q 必须是 [batch, heads, head_dim] 形状")
    if q.shape[0] != cache.batch_size or q.shape[1] != cache.num_heads or q.shape[2] != cache.head_dim:
        raise ValueError("q 的 batch、heads 和 head_dim 必须与 PagedKVCache 对齐")
    if cache.k_quant.original_shape != cache.v_quant.original_shape:
        raise ValueError("cache.k_quant 和 cache.v_quant 的 original_shape 必须一致")
    if len(cache.k_quant.original_shape) != 4:
        raise ValueError("PagedKVCache 的 k_quant/v_quant 必须来自 [physical, heads, block, head_dim] 张量")
    if cache.k_quant.original_shape[2] != cache.block_size:
        raise ValueError("PagedKVCache 物理 block 的 sequence 维必须等于 cache.block_size")
    if cache.k_quant.block_size != cache.v_quant.block_size:
        raise ValueError("K/V 量化 block_size 必须一致")
    if cache.block_size % cache.k_quant.block_size != 0:
        raise ValueError("cache.block_size 必须能被量化 block_size 整除")
    if q.device.type != "cuda":
        raise RuntimeError("Triton paged attention 需要 CUDA q tensor")
    if cache.k_quant.device != q.device or cache.v_quant.device != q.device:
        raise RuntimeError("q 和 cache 的量化 K/V 必须位于同一个 CUDA 设备")
    if cache.block_table.device != q.device or cache.lengths.device != q.device:
        raise RuntimeError("block_table、lengths 必须与 q 位于同一个 CUDA 设备")
    if q.shape[-1] > 256:
        raise ValueError("Triton paged attention 当前只支持 head_dim <= 256")


def _run_paged_quant_attention_triton(
    q: torch.Tensor,
    cache: PagedKVCache,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """运行真正的 Triton Kernel 2，不调用 cache.to_dense()。

    Python 侧只负责整理指针、分配输出和发起 kernel launch。kernel 内部会：
    1. 读取 block_table，完成 logical block -> physical block 的间接寻址。
    2. 从 physical INT8 K/V block 直接 load。
    3. 在寄存器中反量化。
    4. 完成 QK、online softmax 和 PV。

    因此主路径不会 gather 或 materialize 完整 dense KV。
    """

    _validate_triton_paged_inputs(q, cache)

    # contiguous 只整理现有元数据和 INT8 block store 的内存布局；
    # 不会把 paged cache 还原成 dense K/V。
    q_contig = q.contiguous()
    k_values = cache.k_quant.values.contiguous()
    k_scale = cache.k_quant.scale.contiguous()
    k_zero = cache.k_quant.zero_point.contiguous()
    v_values = cache.v_quant.values.contiguous()
    v_scale = cache.v_quant.scale.contiguous()
    v_zero = cache.v_quant.zero_point.contiguous()
    block_table = cache.block_table.contiguous()
    lengths = cache.lengths.contiguous()

    batch, heads, head_dim = q_contig.shape
    max_seq_len = int(cache.max_seq_len)
    table_cols = int(block_table.shape[1])
    physical_quant_blocks = int(k_values.shape[-3])
    page_block_size = int(cache.block_size)
    quant_block_size = int(cache.k_quant.block_size)
    block_d = _next_power_of_2(int(head_dim))

    # 和 Kernel 1 保持一致，每次扫描 64 个逻辑 token。
    block_n = 64
    out = torch.empty_like(q_contig)

    grid = (batch * heads,)
    _paged_quant_attention_kernel[grid](
        q_contig,
        k_values,
        k_scale,
        k_zero,
        v_values,
        v_scale,
        v_zero,
        block_table,
        lengths,
        out,
        heads,
        max_seq_len,
        head_dim,
        table_cols,
        physical_quant_blocks,
        page_block_size,
        quant_block_size,
        1.0 / math.sqrt(float(head_dim)),
        block_n,
        block_d,
        num_warps=4,
    )

    if not return_stats:
        return out

    dense_bytes = 2 * batch * heads * max_seq_len * head_dim * q_contig.element_size()
    quant_bytes = cache.estimated_bytes()
    stats = {
        "dense_kv_bytes": float(dense_bytes),
        "quant_kv_bytes": float(quant_bytes),
        "compression_ratio": float(dense_bytes / max(1, quant_bytes)),
        "physical_blocks": float(cache.block_table.ge(0).sum().item()),
        "materializes_dense_kv": 0.0,
    }
    return out, stats


def fused_dequant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 1 的 Triton 兼容入口：fused dequant attention。

    参数：
    - *_args/**_kwargs: 透传给 attention.fused_dequant_attention 的所有参数。
      当前 wrapper 不重新声明完整签名，是为了和 portable backend 保持兼容，
      同时给未来 Triton kernel 留出替换空间。

    当前行为：
    - 如果安装了 Triton 且输入位于 CUDA，则启动真正的 fused dequant kernel。
    - 否则延迟导入并调用 PyTorch 参考实现，保证 CPU 环境仍可导入和 smoke test。

    设计意图：
    - 让外部代码可以先依赖 `fused_dequant_attention_triton` 这个稳定入口。
    - CUDA 主路径不落地 dense FP16/FP32 K/V，满足 Kernel 1 的核心目标。
    - 没有安装 Triton 时，CPU 测试和基础功能仍能正常跑通。
    """

    if HAS_TRITON and _args and isinstance(_args[0], torch.Tensor) and _args[0].device.type == "cuda":
        return _run_fused_dequant_attention_triton(*_args, **_kwargs)

    # 非 CUDA 或未安装 Triton 时使用 portable fallback；这个 fallback 会 materialize dense KV。
    from .attention import fused_dequant_attention

    # 当前 fallback 直接委托给已测试过的 portable PyTorch 后端。
    return fused_dequant_attention(*_args, **_kwargs)


def paged_quant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 2 的 Triton 兼容入口：paged quant KV attention。

    参数：
    - *_args/**_kwargs: 透传给 attention.paged_quant_attention 的所有参数。

    当前行为：
    - 如果安装了 Triton，且 q/cache 都位于 CUDA，则启动真正的 paged Triton kernel。
    - 否则委托给 PyTorch 参考实现 paged_quant_attention，保证 CPU 环境可用。

    Triton 主路径：
    - kernel 内部直接读取 block_table，按 logical block 间接寻址 physical KV block。
    - 直接读取 INT8 K/V block。
    - 融合 INT8 反量化、QK、softmax 和 PV，避免物化 dense KV。
    """

    if (
        HAS_TRITON
        and _args
        and isinstance(_args[0], torch.Tensor)
        and _args[0].device.type == "cuda"
        and len(_args) >= 2
        and isinstance(_args[1], PagedKVCache)
    ):
        return _run_paged_quant_attention_triton(*_args, **_kwargs)

    # 非 CUDA 或未安装 Triton 时使用 portable fallback；这个 fallback 会调用 cache.to_dense()。
    from .attention import paged_quant_attention

    # 当前 fallback 直接复用 PyTorch 参考实现，保证语义和测试结果一致。
    return paged_quant_attention(*_args, **_kwargs)
