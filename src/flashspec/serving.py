from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import torch

from .attention import paged_quant_attention
from .paged import PagedKVCache
from .runtime import device_name, resolve_device, resolve_dtype, synchronize


@dataclass(frozen=True)
class ServingConfig:
    requests: int = 8
    prompt_len: int = 128
    decode_steps: int = 16
    heads: int = 8
    head_dim: int = 64
    block_size: int = 16
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"


def run_decode_simulation(config: ServingConfig) -> Dict[str, float]:
    """Run a deterministic decode-loop simulation over the paged KV path."""

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    shape = (config.requests, config.heads, config.prompt_len, config.head_dim)
    k = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    synchronize(device)
    start = perf_counter()
    cache = PagedKVCache.from_dense(k, v, block_size=config.block_size)
    q = torch.randn((config.requests, config.heads, config.head_dim), generator=generator, device=device, dtype=dtype)
    _ = paged_quant_attention(q, cache)
    synchronize(device)
    ttft_ms = (perf_counter() - start) * 1000.0

    synchronize(device)
    decode_start = perf_counter()
    for _step in range(config.decode_steps):
        q = torch.randn((config.requests, config.heads, config.head_dim), generator=generator, device=device, dtype=dtype)
        out = paged_quant_attention(q, cache)
        next_k = out.unsqueeze(2).to(k.dtype)
        next_v = torch.tanh(out).unsqueeze(2).to(v.dtype)
        cache = cache.append(next_k, next_v)
    synchronize(device)
    decode_ms = (perf_counter() - decode_start) * 1000.0

    generated = config.requests * config.decode_steps
    return {
        "device": str(device),
        "device_name": device_name(device),
        "dtype": str(dtype).replace("torch.", ""),
        "requests": float(config.requests),
        "prompt_len": float(config.prompt_len),
        "decode_steps": float(config.decode_steps),
        "ttft_ms": ttft_ms,
        "tpot_ms": decode_ms / max(1, config.decode_steps),
        "tokens_per_second": generated / max(1.0e-9, decode_ms / 1000.0),
    }
