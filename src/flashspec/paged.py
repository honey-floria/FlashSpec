from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .quant import QuantizedTensor, dequantize_int8_per_block, estimate_quantized_bytes, quantize_int8_per_block


@dataclass(frozen=True)
class PagedKVCache:
    """Paged INT8 KV cache.

    `block_table[b, logical_block]` maps a sequence block to a physical block.
    Quantized physical blocks are stored as `[physical, heads, block, dim]`
    before quantization.
    """

    k_quant: QuantizedTensor
    v_quant: QuantizedTensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    block_size: int
    max_seq_len: int

    @property
    def batch_size(self) -> int:
        return int(self.block_table.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.k_quant.original_shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.k_quant.original_shape[-1])

    @classmethod
    def from_dense(cls, k: torch.Tensor, v: torch.Tensor, block_size: int = 16) -> "PagedKVCache":
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError("k and v must have shape [batch, heads, seq_len, head_dim]")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        batch, heads, seq_len, head_dim = k.shape
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = batch * blocks_per_seq
        padded_seq_len = blocks_per_seq * block_size
        if padded_seq_len != seq_len:
            pad_shape = (batch, heads, padded_seq_len - seq_len, head_dim)
            k = torch.cat([k, k.new_zeros(pad_shape)], dim=2)
            v = torch.cat([v, v.new_zeros(pad_shape)], dim=2)

        physical_k = (
            k.reshape(batch, heads, blocks_per_seq, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(total_blocks, heads, block_size, head_dim)
        )
        physical_v = (
            v.reshape(batch, heads, blocks_per_seq, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(total_blocks, heads, block_size, head_dim)
        )
        block_table = torch.arange(total_blocks, dtype=torch.int64, device=k.device).reshape(batch, blocks_per_seq)

        return cls(
            k_quant=quantize_int8_per_block(physical_k, block_size=block_size),
            v_quant=quantize_int8_per_block(physical_v, block_size=block_size),
            block_table=block_table,
            lengths=torch.full((batch,), seq_len, dtype=torch.int64, device=k.device),
            block_size=block_size,
            max_seq_len=seq_len,
        )

    def to_dense(self) -> Tuple[torch.Tensor, torch.Tensor]:
        physical_k = dequantize_int8_per_block(self.k_quant, dtype=torch.float32)
        physical_v = dequantize_int8_per_block(self.v_quant, dtype=torch.float32)
        table = self.block_table.clamp_min(0)
        gathered_k = physical_k[table]
        gathered_v = physical_v[table]
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

    def estimated_bytes(self) -> int:
        table_bytes = self.block_table.numel() * self.block_table.element_size()
        length_bytes = self.lengths.numel() * self.lengths.element_size()
        return estimate_quantized_bytes(self.k_quant) + estimate_quantized_bytes(self.v_quant) + table_bytes + length_bytes
