from __future__ import annotations

from typing import Any

import math
import os

import torch

from .quant import QuantizedTensor, estimate_quantized_bytes
from .triton_utils import HAS_TRITON, next_power_of_2, tl, triton


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

        与单 kernel 的唯一区别：每个 program 只扫自己那段 token
        `[split_id*chunk_tokens, (split_id+1)*chunk_tokens)`，产出该段的
        online-softmax 部分状态 (partial_m, partial_l, partial_acc) 写入 scratch，
        不直接写 out。合并交给 _combine_splits_kernel。

        chunk_tokens 由 launcher 保证是 block_n 的整数倍，因此内层循环覆盖的范围
        恰好等于本段 token 区间，尾部用 effective_len mask 处理不整除。
        """

        pid = tl.program_id(0)
        split_id = tl.program_id(1)

        batch_idx = pid // heads
        head_idx = pid - batch_idx * heads

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        q_base = (batch_idx * heads + head_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        effective_len = seq_len
        if has_lengths:
            loaded_len = tl.load(lengths_ptr + batch_idx)
            effective_len = tl.minimum(loaded_len, seq_len)

        # 本段负责的 token 区间。split_end 不超过 seq_len。
        split_start = split_id * chunk_tokens
        split_end = tl.minimum(split_start + chunk_tokens, seq_len)

        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 只扫本段。chunk_tokens 是 block_n 的整数倍，循环次数编译期已知。
        for local in range(0, chunk_tokens, block_n):
            offs_s = split_start + local + tl.arange(0, block_n)
            s_mask = (offs_s < effective_len) & (offs_s < split_end)

            quant_block = offs_s // block_size
            block_offset = offs_s - quant_block * block_size

            kv_offsets = (
                ((((batch_idx * heads + head_idx) * n_blocks + quant_block[:, None]) * block_size + block_offset[:, None])
                 * head_dim)
                + offs_d[None, :]
            )
            kv_mask = s_mask[:, None] & d_mask[None, :]

            qparam_offsets = (batch_idx * heads + head_idx) * n_blocks + quant_block
            k_scale = tl.load(k_scale_ptr + qparam_offsets, mask=s_mask, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_point_ptr + qparam_offsets, mask=s_mask, other=0).to(tl.float32)
            k_i8 = tl.load(k_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
            k_deq = (k_i8 + 128.0 - k_zero[:, None]) * k_scale[:, None]

            scores = tl.sum(k_deq * q[None, :], axis=1) * sm_scale
            scores = tl.where(s_mask, scores, -3.4028234663852886e38)

            block_m = tl.max(scores, axis=0)
            new_m = tl.maximum(m, block_m)
            old_scale = tl.exp(m - new_m)
            probs = tl.exp(scores - new_m)
            probs = tl.where(s_mask, probs, 0.0)
            new_l = l * old_scale + tl.sum(probs, axis=0)

            v_scale = tl.load(v_scale_ptr + qparam_offsets, mask=s_mask, other=1.0).to(tl.float32)
            v_zero = tl.load(v_zero_point_ptr + qparam_offsets, mask=s_mask, other=0).to(tl.float32)
            v_i8 = tl.load(v_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
            v_deq = (v_i8 + 128.0 - v_zero[:, None]) * v_scale[:, None]

            acc = acc * old_scale + tl.sum(probs[:, None] * v_deq, axis=0)
            m = new_m
            l = new_l

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

        每个 program 负责一个 [batch, head]，读取自己 S 段的 (partial_m, partial_l,
        partial_acc)，做跨段 online-softmax rescale。数学与 kernel 内跨 block 的
        合并完全同构，只是这里跨的是 split 段。
        """

        pid = tl.program_id(0)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 逐段合并。空段的 partial_m=-FLT_MAX、partial_l=0，rescale 后贡献为 0，
        # 不会污染结果；全空时 l 保持 0，最后走除零保护输出 0。
        for s in range(0, num_splits):
            idx = pid * num_splits + s
            pm = tl.load(partial_m_ptr + idx)
            pl = tl.load(partial_l_ptr + idx)
            pacc = tl.load(partial_acc_ptr + idx * head_dim + offs_d, mask=d_mask, other=0.0)

            new_m = tl.maximum(m, pm)
            old_scale = tl.exp(m - new_m)
            cur_scale = tl.exp(pm - new_m)
            l = l * old_scale + pl * cur_scale
            acc = acc * old_scale + pacc * cur_scale
            m = new_m

        denom = tl.where(l > 0.0, l, 1.0)
        out = acc / denom
        out = tl.where(l > 0.0, out, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out, mask=d_mask)


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


# Split-K 自适应参数（见 doc/split-k-plan.md）。
# TOKENS_PER_SPLIT：每段目标 token 数；S_MAX：段数上限，防止 scratch 过大。
_TOKENS_PER_SPLIT = 512
_S_MAX = 32


def _compute_num_splits(seq_len: int, block_n: int) -> int:
    """自适应 S = clamp(ceil(seq_len / TOKENS_PER_SPLIT), 1, S_MAX)。

    S==1 时退化回单 kernel 快路径（零 combine 开销）。短序列(512)→1，
    2048→4，4096→8。段太多不划算，故用 S_MAX 封顶。

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

    if seq_len <= _TOKENS_PER_SPLIT:
        return 1
    s = (seq_len + _TOKENS_PER_SPLIT - 1) // _TOKENS_PER_SPLIT
    return max(1, min(s, _S_MAX))


def _resolve_block_n() -> int:
    """每个 program 每轮扫描的 token 数。默认 64。

    实验 4（降寄存器）A/B 开关：k_deq/v_deq 临时 tile 是 [block_n, block_d]，
    寄存器占用与 block_n 成正比。调小 block_n 可降低 registers_per_thread、
    抬高占用率天花板，代价是循环轮数增加。用环境变量隔离该变量：
    - FLASHSPEC_BLOCK_N=32  每轮扫 32 token（更少寄存器）；
    - FLASHSPEC_BLOCK_N=64  默认；
    - 未设置或非法值        用默认 64。
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
    return 64


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
    block_d = next_power_of_2(int(head_dim))

    # block_n 是每个 program 每轮扫描的 token 数（实验 4：可用 FLASHSPEC_BLOCK_N 覆盖）。
    # tile [block_n, block_d] 越小，registers_per_thread 越低、占用率天花板越高。
    block_n = _resolve_block_n()
    num_warps = _resolve_num_warps()
    out = torch.empty_like(q_contig)
    sm_scale = 1.0 / math.sqrt(float(head_dim))

    num_splits = _compute_num_splits(seq_len, block_n)

    if num_splits == 1:
        # S==1 快路径：直接用原单 kernel 写 out，不分配 scratch、不启 combine。
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
            sm_scale,
            lengths is not None,
            block_n,
            block_d,
            num_warps=num_warps,
        )
    else:
        # Split-K：把 seq_len 切成 num_splits 段并行，grid=(batch*heads, S)。
        # chunk_tokens 向上取整到 block_n 的整数倍，使 split kernel 内层循环恰好
        # 覆盖本段，尾段用 mask 处理不整除。
        chunk_tokens = ((seq_len + num_splits - 1) // num_splits + block_n - 1) // block_n * block_n
        rows = batch * heads
        # scratch：partial 状态，不是 dense KV。s4096/d128,S=8 时 acc ≈ 16.8MB。
        partial_m = torch.empty((rows, num_splits), device=q_contig.device, dtype=torch.float32)
        partial_l = torch.empty((rows, num_splits), device=q_contig.device, dtype=torch.float32)
        partial_acc = torch.empty((rows, num_splits, head_dim), device=q_contig.device, dtype=torch.float32)

        _fused_dequant_attention_split_kernel[(rows, num_splits)](
            q_contig,
            k_values,
            k_scale,
            k_zero,
            v_values,
            v_scale,
            v_zero,
            lengths_contig,
            partial_m,
            partial_l,
            partial_acc,
            heads,
            seq_len,
            head_dim,
            n_blocks,
            block_size,
            sm_scale,
            lengths is not None,
            num_splits,
            chunk_tokens,
            block_n,
            block_d,
            num_warps=num_warps,
        )
        _combine_splits_kernel[(rows,)](
            partial_m,
            partial_l,
            partial_acc,
            out,
            num_splits,
            head_dim,
            block_d,
            num_warps=num_warps,
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

    当前行为：
    - 如果安装了 Triton 且输入位于 CUDA，则启动真正的 fused dequant kernel。
    - 否则延迟导入并调用 PyTorch 参考实现，保证 CPU 环境仍可导入和 smoke test。
    """

    if HAS_TRITON and _args and isinstance(_args[0], torch.Tensor) and _args[0].device.type == "cuda":
        return _run_fused_dequant_attention_triton(*_args, **_kwargs)

    # 非 CUDA 或未安装 Triton 时使用 portable fallback；这个 fallback 会 materialize dense KV。
    from .attention import fused_dequant_attention

    return fused_dequant_attention(*_args, **_kwargs)
