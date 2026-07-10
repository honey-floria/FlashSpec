from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flashspec import (
    PagedKVCache,
    fused_dequant_attention,
    fused_dequant_attention_triton,
    paged_quant_attention,
    paged_quant_attention_triton,
    quantize_int8_per_block,
    reference_attention,
)
from flashspec.runtime import device_name, resolve_device, resolve_dtype, synchronize
from flashspec.triton_kernels import HAS_TRITON


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashSpec decode attention microbenchmark")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--backend", choices=["dense", "fused", "triton_fused", "paged", "triton_paged"], default="paged")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    q = torch.randn((args.batch, args.heads, args.head_dim), generator=generator, device=device, dtype=dtype)
    k = torch.randn((args.batch, args.heads, args.seq_len, args.head_dim), generator=generator, device=device, dtype=dtype)
    v = torch.randn((args.batch, args.heads, args.seq_len, args.head_dim), generator=generator, device=device, dtype=dtype)

    if args.backend == "dense":
        def run() -> torch.Tensor:
            return reference_attention(q, k, v)

        stats = {
            "dense_kv_bytes": float(2 * k.numel() * k.element_size()),
            "quant_kv_bytes": float(2 * k.numel() * k.element_size()),
            "compression_ratio": 1.0,
            "materializes_dense_kv": 0.0,
        }
    elif args.backend == "fused":
        kq = quantize_int8_per_block(k, block_size=args.block_size)
        vq = quantize_int8_per_block(v, block_size=args.block_size)

        def run() -> torch.Tensor:
            return fused_dequant_attention(q, kq, vq)

        _, stats = fused_dequant_attention(q, kq, vq, return_stats=True)
    elif args.backend == "triton_fused":
        if not HAS_TRITON:
            raise RuntimeError("triton_fused backend 需要安装 Triton：python -m pip install -e \".[triton]\"")
        if device.type != "cuda":
            raise RuntimeError("triton_fused backend 需要 CUDA 设备，请使用 --device cuda 或 --device auto")

        kq = quantize_int8_per_block(k, block_size=args.block_size)
        vq = quantize_int8_per_block(v, block_size=args.block_size)

        def run() -> torch.Tensor:
            return fused_dequant_attention_triton(q, kq, vq)

        _, stats = fused_dequant_attention_triton(q, kq, vq, return_stats=True)
    elif args.backend == "paged":
        cache = PagedKVCache.from_dense(k, v, block_size=args.block_size)

        def run() -> torch.Tensor:
            return paged_quant_attention(q, cache)

        _, stats = paged_quant_attention(q, cache, return_stats=True)
    else:
        if not HAS_TRITON:
            raise RuntimeError("triton_paged backend 需要安装 Triton：python -m pip install -e \".[triton]\"")
        if device.type != "cuda":
            raise RuntimeError("triton_paged backend 需要 CUDA 设备，请使用 --device cuda 或 --device auto")

        cache = PagedKVCache.from_dense(k, v, block_size=args.block_size)

        def run() -> torch.Tensor:
            return paged_quant_attention_triton(q, cache)

        _, stats = paged_quant_attention_triton(q, cache, return_stats=True)

    for _ in range(2):
        run()

    synchronize(device)
    start = perf_counter()
    for _ in range(args.iters):
        out = run()
    synchronize(device)
    elapsed = perf_counter() - start
    latency_ms = elapsed * 1000.0 / max(1, args.iters)
    result = {
        "backend": args.backend,
        "device": str(device),
        "device_name": device_name(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch": args.batch,
        "heads": args.heads,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "iters": args.iters,
        "latency_ms": latency_ms,
        "tokens_per_second": args.batch / max(1.0e-9, latency_ms / 1000.0),
        "estimated_dense_kv_bytes": stats["dense_kv_bytes"],
        "estimated_quant_kv_bytes": stats["quant_kv_bytes"],
        "compression_ratio": stats["compression_ratio"],
        "materializes_dense_kv": bool(stats["materializes_dense_kv"]),
        "bandwidth_fields_are_estimates": True,
        "effective_dense_kv_bandwidth_gbps": stats["dense_kv_bytes"] / max(1.0e-9, latency_ms / 1000.0) / 1.0e9,
        "effective_quant_kv_bandwidth_gbps": stats["quant_kv_bytes"] / max(1.0e-9, latency_ms / 1000.0) / 1.0e9,
        "output_checksum": float(out.float().sum().item()),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
