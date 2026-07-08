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

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "PagedKVCache":
        """Append dense KV tokens to the paged cache.

        This portable implementation updates physical blocks and requantizes the
        block store. It avoids rebuilding from a full dense `[batch, heads, seq,
        dim]` tensor, while keeping the public behavior deterministic on CPU and
        CUDA.
        """

        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError("k_new and v_new must have shape [batch, heads, tokens, head_dim]")
        if k_new.shape != v_new.shape:
            raise ValueError("k_new and v_new must have the same shape")
        if k_new.shape[0] != self.batch_size or k_new.shape[1] != self.num_heads or k_new.shape[3] != self.head_dim:
            raise ValueError("new KV shape must match cache batch, heads, and head_dim")
        append_tokens = int(k_new.shape[2])
        if append_tokens <= 0:
            raise ValueError("append token dimension must be non-empty")
        if k_new.device != self.block_table.device or v_new.device != self.block_table.device:
            raise ValueError("new KV tensors must be on the same device as the cache")

        physical_k = dequantize_int8_per_block(self.k_quant, dtype=torch.float32).clone()
        physical_v = dequantize_int8_per_block(self.v_quant, dtype=torch.float32).clone()
        table = self.block_table.clone()
        lengths = self.lengths.clone()

        required_blocks = ((lengths + append_tokens + self.block_size - 1) // self.block_size).max()
        required_cols = int(required_blocks.item())
        if required_cols > table.shape[1]:
            pad = table.new_full((self.batch_size, required_cols - table.shape[1]), -1)
            table = torch.cat([table, pad], dim=1)

        k_new = k_new.to(dtype=torch.float32)
        v_new = v_new.to(dtype=torch.float32)
        for batch_idx in range(self.batch_size):
            start = int(lengths[batch_idx].item())
            for token_idx in range(append_tokens):
                position = start + token_idx
                logical_block = position // self.block_size
                block_offset = position % self.block_size
                physical_idx = int(table[batch_idx, logical_block].item())
                if physical_idx < 0:
                    physical_idx = int(physical_k.shape[0])
                    empty_shape = (1, self.num_heads, self.block_size, self.head_dim)
                    physical_k = torch.cat([physical_k, physical_k.new_zeros(empty_shape)], dim=0)
                    physical_v = torch.cat([physical_v, physical_v.new_zeros(empty_shape)], dim=0)
                    table[batch_idx, logical_block] = physical_idx
                physical_k[physical_idx, :, block_offset, :] = k_new[batch_idx, :, token_idx, :]
                physical_v[physical_idx, :, block_offset, :] = v_new[batch_idx, :, token_idx, :]

        lengths = lengths + append_tokens
        return PagedKVCache(
            k_quant=quantize_int8_per_block(physical_k, block_size=self.block_size),
            v_quant=quantize_int8_per_block(physical_v, block_size=self.block_size),
            block_table=table,
            lengths=lengths,
            block_size=self.block_size,
            max_seq_len=int(lengths.max().item()),
        )

    def estimated_bytes(self) -> int:
        table_bytes = self.block_table.numel() * self.block_table.element_size()
        length_bytes = self.lengths.numel() * self.lengths.element_size()
        return estimate_quantized_bytes(self.k_quant) + estimate_quantized_bytes(self.v_quant) + table_bytes + length_bytes
