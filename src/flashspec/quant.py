from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class QuantizedTensor:
    """Affine INT8 tensor quantized per contiguous sequence block."""

    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    block_size: int
    original_shape: Tuple[int, ...]

    @property
    def device(self) -> torch.device:
        return self.values.device


def _validate_block_tensor(x: torch.Tensor, block_size: int) -> None:
    if x.ndim < 2:
        raise ValueError("expected tensor with at least sequence and feature dimensions")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if x.shape[-1] <= 0 or x.shape[-2] <= 0:
        raise ValueError("sequence and feature dimensions must be non-empty")


def _pad_sequence_dim(x: torch.Tensor, block_size: int) -> torch.Tensor:
    seq_len = x.shape[-2]
    pad = (block_size - seq_len % block_size) % block_size
    if pad == 0:
        return x
    pad_shape = (*x.shape[:-2], pad, x.shape[-1])
    return torch.cat([x, x.new_zeros(pad_shape)], dim=-2)


def quantize_int8_per_block(x: torch.Tensor, block_size: int = 16) -> QuantizedTensor:
    """Quantize `x` using affine INT8 parameters per sequence block.

    The sequence dimension is `-2`, and the feature/head dimension is `-1`.
    Scales and zero-points are computed over `(block_size, feature_dim)` for
    every leading index. Values are stored as signed int8, while zero-points are
    in the shifted uint8 domain used by `values + 128`.
    """

    _validate_block_tensor(x, block_size)
    original_shape = tuple(x.shape)
    x_float = _pad_sequence_dim(x.detach().to(torch.float32), block_size)
    leading = x_float.shape[:-2]
    seq_len = x_float.shape[-2]
    head_dim = x_float.shape[-1]
    n_blocks = seq_len // block_size
    blocked = x_float.reshape(*leading, n_blocks, block_size, head_dim)

    block_min = blocked.amin(dim=(-2, -1), keepdim=True)
    block_max = blocked.amax(dim=(-2, -1), keepdim=True)
    scale = ((block_max - block_min) / 255.0).clamp_min(1.0e-8)
    zero_point = torch.round(-block_min / scale).clamp(0, 255)

    q_uint8 = torch.round(blocked / scale + zero_point).clamp(0, 255)
    q_int8 = (q_uint8.to(torch.int16) - 128).to(torch.int8)
    return QuantizedTensor(
        values=q_int8,
        scale=scale,
        zero_point=zero_point.to(torch.int16),
        block_size=block_size,
        original_shape=original_shape,
    )


def dequantize_int8_per_block(q: QuantizedTensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dequantize a tensor produced by `quantize_int8_per_block`."""

    q_uint8 = q.values.to(torch.int16) + 128
    x = (q_uint8.to(torch.float32) - q.zero_point.to(torch.float32)) * q.scale.to(torch.float32)
    leading = q.original_shape[:-2]
    seq_len = q.original_shape[-2]
    head_dim = q.original_shape[-1]
    x = x.reshape(*leading, -1, head_dim)[..., :seq_len, :]
    return x.to(dtype=dtype) if dtype is not None else x


def estimate_quantized_bytes(q: QuantizedTensor) -> int:
    """Return an approximate storage footprint for a quantized tensor."""

    return q.values.numel() + q.scale.numel() * 4 + q.zero_point.numel() * 2

