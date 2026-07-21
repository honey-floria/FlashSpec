from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .paged import PagedKVCache
from .quant import (
    QuantizedTensor,
    build_compression_stats,
    dequantize_int8_per_block,
    estimate_quantized_bytes,
)


def _validate_decode_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """校验 decode attention 输入张量的形状。

    参数：
    - q: 当前 decode step 的 query，形状必须是 [batch, heads, head_dim]。
      decode 阶段一次只处理每条序列的一个新 token，所以 q 没有 seq_len 维度。
    - k/v: 历史 KV cache，形状必须是 [batch, heads, seq_len, head_dim]。

    这个函数只做形状一致性检查，不检查 dtype、device 或数值范围。
    """

    # q 必须是 3 维：[batch, heads, head_dim]。
    if q.ndim != 3:
        raise ValueError("q 必须是 [batch, heads, head_dim] 形状")

    # k/v 必须是 4 维：[batch, heads, seq_len, head_dim]。
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k 和 v 必须是 [batch, heads, seq_len, head_dim] 形状")

    # K 和 V 代表同一批历史 token 的 key/value，形状必须完全一致。
    if k.shape != v.shape:
        raise ValueError("k 和 v 的形状必须一致")

    # q 的 batch、heads、head_dim 需要分别匹配 k/v 的对应维度。
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[2] != k.shape[3]:
        raise ValueError("q 的 batch、heads 和 head_dim 必须与 k/v 对齐")


def _sequence_mask(seq_len: int, lengths: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    """根据每条序列的有效长度生成 padding mask。

    参数：
    - seq_len: dense K/V 的 sequence 维长度。
    - lengths: 可选的 [batch] 张量，每个元素表示对应序列的有效 token 数。
    - device: mask 所在设备，需要和 attention scores 所在设备一致。

    返回的 mask 形状为 [batch, seq_len]。值为 True 的位置表示该 token
    已经超出当前序列的有效长度，需要在 attention scores 中被屏蔽。
    """

    # 如果没有传入 lengths，说明 K/V 的所有 seq_len 位置都有效，不需要 mask。
    if lengths is None:
        return None

    # positions: [1, seq_len]，表示每个 sequence 位置的下标。
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    # lengths.to(device).unsqueeze(1): [batch, 1]。
    # 广播比较后得到 [batch, seq_len]：
    # position >= length 的位置为 True，表示 padding 或未使用位置。
    return positions >= lengths.to(device).unsqueeze(1)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """单 query token 的 decode attention 参考实现。

    参数形状：
    - q: [batch, heads, head_dim]，每条序列当前要解码的 query token。
    - k/v: [batch, heads, seq_len, head_dim]，历史 KV cache。
    - lengths: 可选的 [batch] 有效长度，用于屏蔽 padding 或未使用位置。

    计算流程：
    1. 用 q 和 k 做缩放点积，得到 [batch, heads, seq_len] 的 scores。
    2. 如果传入 lengths，则把超出有效长度的位置置为极小值。
    3. 对 scores 做 softmax。
    4. 用 softmax 权重加权求和 v，得到 [batch, heads, head_dim]。

    该函数用于正确性对齐，不负责模拟真正高性能 kernel 的内存访问模式。
    """

    # 先校验 q/k/v 的基本布局，避免 einsum 报出难读的低层错误。
    _validate_decode_shapes(q, k, v)

    # scale: scaled dot-product attention 中的 1 / sqrt(head_dim)。
    # 这个缩放用于避免 head_dim 较大时 dot product 数值过大，导致 softmax 过尖。
    scale = 1.0 / math.sqrt(q.shape[-1])

    # scores: [batch, heads, seq_len]。
    # einsum 维度含义：
    # - q: "bhd" = batch、head、head_dim
    # - k: "bhsd" = batch、head、seq_len、head_dim
    # - 输出 "bhs" = 每个 query 对历史每个 key 的注意力分数
    # q/k 转成 float32 是为了让参考实现的数值更稳定。
    scores = torch.einsum("bhd,bhsd->bhs", q.to(torch.float32), k.to(torch.float32)) * scale

    # mask: None 或 [batch, seq_len]。
    # k.shape[-2] 是 seq_len，也就是历史 KV cache 的 token 维度。
    mask = _sequence_mask(k.shape[-2], lengths, k.device)
    if mask is not None:
        # scores 的形状是 [batch, heads, seq_len]，
        # mask[:, None, :] 插入 heads 维后可广播到所有 attention heads。
        # 被 mask 的位置填成当前 dtype 可表示的最小值，使 softmax 后概率接近 0。
        scores = scores.masked_fill(mask[:, None, :], torch.finfo(scores.dtype).min)

    # probs: [batch, heads, seq_len]，表示每个 query token 对所有历史 token 的权重。
    probs = torch.softmax(scores, dim=-1)

    # 输出形状为 [batch, heads, head_dim]。
    # v 转 float32 参与加权求和，最后再转回 q 的 dtype，保持 API 输出类型和输入 q 一致。
    return torch.einsum("bhs,bhsd->bhd", probs, v.to(torch.float32)).to(q.dtype)


def fused_dequant_attention(
    q: torch.Tensor,
    k_quant: QuantizedTensor,
    v_quant: QuantizedTensor,
    lengths: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """Kernel 1 API：反量化 INT8 KV，并执行 decode attention。

    当前可移植 PyTorch 后端会先把量化 KV 还原成 dense float32 KV，
    再调用 reference_attention，以保证行为清晰且便于测试正确性。

    真正的 CUDA/Triton 部署可以保留同样 API，在 kernel 内融合
    INT8 反量化和 attention 计算，避免物化完整 dense KV。

    如果 return_stats=True，返回 (out, stats)，其中 stats 包含：
    - dense_kv_bytes: 如果以 dense 形式保存 KV，估算需要的字节数。
    - quant_kv_bytes: 当前 INT8 量化 KV 的估算字节数。
    - compression_ratio: dense KV 相对 quant KV 的压缩比例。
    - materializes_dense_kv: 当前后端是否会物化 dense KV；此处为 1.0。
    """

    # k_quant/v_quant 是 per-block INT8 表示。当前参考后端先显式反量化成 dense float32。
    # 真实融合 kernel 会在读取每个 INT8 元素时即时反量化，而不是生成完整 dense k/v。
    k = dequantize_int8_per_block(k_quant, dtype=torch.float32)
    v = dequantize_int8_per_block(v_quant, dtype=torch.float32)

    # 复用 dense reference attention，确保 quant 路径的 attention 语义和基准一致。
    out = reference_attention(q, k, v, lengths=lengths)

    # 默认只返回 attention 输出；benchmark 需要时才计算统计信息。
    if not return_stats:
        return out

    # dense_bytes: 如果 K/V 都以 q.dtype 的 dense tensor 保存，大约需要多少字节。
    # 这里乘以 2 是因为 K 和 V 各一份；k.numel() 与 v.numel() 相同。
    dense_bytes = 2 * k.numel() * q.element_size()

    # quant_bytes: 量化后 K/V 的估算存储，包括 INT8 values、scale 和 zero_point。
    quant_bytes = estimate_quantized_bytes(k_quant) + estimate_quantized_bytes(v_quant)

    # stats: 给 benchmark/report 使用的轻量指标，不参与 attention 正确性计算。
    # 当前 PyTorch 参考后端会物化 dense KV，所以 materializes_dense_kv=True。
    stats = build_compression_stats(dense_bytes, quant_bytes, materializes_dense_kv=True)
    return out, stats


def paged_quant_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """Kernel 2 API：基于 paged quantized KV cache 执行 decode attention。

    PagedKVCache 使用 block_table 将每条序列的逻辑 block 映射到物理
    KV block。当前 PyTorch 参考实现会先调用 cache.to_dense() 还原出
    dense K/V，再基于 cache.lengths 做有效长度 mask，并复用
    reference_attention 完成计算。

    真正的高性能后端可以保留同样 API，直接在 kernel 中根据 block_table
    间接寻址物理 block，同时融合 INT8 反量化和 attention 计算。

    如果 return_stats=True，返回 (out, stats)，其中 stats 包含：
    - dense_kv_bytes: 如果以 dense 形式保存 KV，估算需要的字节数。
    - quant_kv_bytes: paged quant KV cache 的估算字节数。
    - compression_ratio: dense KV 相对 paged quant KV 的压缩比例。
    - physical_blocks: block_table 中有效物理 block 的数量。
    - materializes_dense_kv: 当前后端是否会物化 dense KV；此处为 1.0。
    """

    # cache.to_dense() 会：
    # 1. 反量化物理 K/V block；
    # 2. 按 block_table 收集每条序列的逻辑 block；
    # 3. 还原为 [batch, heads, max_seq_len, head_dim] 的 dense K/V。
    k, v = cache.to_dense()

    # 使用 cache.lengths 屏蔽每条序列中超出真实有效长度的位置。
    # 这对最后一个未填满的 block 以及尚未分配的 block 都很重要。
    out = reference_attention(q, k, v, lengths=cache.lengths)

    # 默认只返回 attention 输出；统计信息仅在 benchmark 或调试时需要。
    if not return_stats:
        return out

    # dense_bytes: 如果 paged cache 被还原成 dense KV，并按 q.dtype 保存，估算需要多少字节。
    dense_bytes = 2 * k.numel() * q.element_size()

    # quant_bytes: paged quant cache 的估算存储，包括 K/V 量化块、block_table 和 lengths。
    quant_bytes = cache.estimated_bytes()

    # stats: paged quant attention 路径的压缩和实现特征指标。
    # physical_blocks 是 block_table >= 0 的条目数（已分配/有效的物理 block 引用数）；
    # 当前 PyTorch 参考实现仍会通过 cache.to_dense() 物化 dense KV。
    stats = build_compression_stats(
        dense_bytes,
        quant_bytes,
        materializes_dense_kv=True,
        physical_blocks=float(cache.block_table.ge(0).sum().item()),
    )
    return out, stats
