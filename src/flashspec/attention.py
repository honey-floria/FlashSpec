from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .paged import PagedKVCache
from .quant import QuantizedTensor, dequantize_int8_per_block, estimate_quantized_bytes


def _validate_decode_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, heads, head_dim]")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have shape [batch, heads, seq_len, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v shapes must match")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[2] != k.shape[3]:
        raise ValueError("q shape must match k/v batch, heads, and head_dim")


def _sequence_mask(seq_len: int, lengths: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if lengths is None:
        return None
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    return positions >= lengths.to(device).unsqueeze(1)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference decode attention for one query token per sequence."""

    _validate_decode_shapes(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhd,bhsd->bhs", q.to(torch.float32), k.to(torch.float32)) * scale
    mask = _sequence_mask(k.shape[-2], lengths, k.device)
    if mask is not None:
        scores = scores.masked_fill(mask[:, None, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", probs, v.to(torch.float32)).to(q.dtype)


def fused_dequant_attention(
    q: torch.Tensor,
    k_quant: QuantizedTensor,
    v_quant: QuantizedTensor,
    lengths: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """Kernel 1 API: dequantize INT8 KV and run decode attention.

    The portable PyTorch backend materializes dequantized KV for correctness.
    CUDA/Triton deployments can replace this function through the same API.
    """

    k = dequantize_int8_per_block(k_quant, dtype=torch.float32)
    v = dequantize_int8_per_block(v_quant, dtype=torch.float32)
    out = reference_attention(q, k, v, lengths=lengths)
    if not return_stats:
        return out
    dense_bytes = 2 * k.numel() * 2
    quant_bytes = estimate_quantized_bytes(k_quant) + estimate_quantized_bytes(v_quant)
    stats = {
        "dense_kv_bytes": float(dense_bytes),
        "quant_kv_bytes": float(quant_bytes),
        "compression_ratio": float(dense_bytes / max(1, quant_bytes)),
    }
    return out, stats


def paged_quant_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
    """Kernel 2 API: decode attention through paged quantized KV blocks."""

    k, v = cache.to_dense()
    out = reference_attention(q, k, v, lengths=cache.lengths)
    if not return_stats:
        return out
    dense_bytes = 2 * k.numel() * 2
    quant_bytes = cache.estimated_bytes()
    stats = {
        "dense_kv_bytes": float(dense_bytes),
        "quant_kv_bytes": float(quant_bytes),
        "compression_ratio": float(dense_bytes / max(1, quant_bytes)),
        "physical_blocks": float(cache.block_table.ge(0).sum().item()),
    }
    return out, stats

