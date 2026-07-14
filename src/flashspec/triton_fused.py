from __future__ import annotations

from typing import Any

import math

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
    block_d = next_power_of_2(int(head_dim))

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
        # num_warps=8：grid 只有 batch*heads(=512) 个 program，A100 有 108 个 SM，
        # 单 program 4 warps 时每 SM 仅 ~19 warps（occupancy ~25%）。提到 8 warps
        # 可把 occupancy 拉到 ~59%，增加并发访存请求以更好地打满 HBM 带宽。
        num_warps=8,
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
