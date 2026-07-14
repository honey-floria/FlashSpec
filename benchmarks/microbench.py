from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

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
from flashspec.ncu_parse import parse_ncu_csv

# 回填 measured_* 字段所需的最小 ncu metric 集合（与 ncu_parse 对齐）。
NCU_METRICS = (
    "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
)
# 默认只 profile 这么多次 launch（warmup 之后），控制 ncu 开销。
NCU_LAUNCH_COUNT = 5


def _percentile(values: list[float], percentile: float) -> float:
    """计算 latency 百分位数。

    使用线性插值，避免只在 numpy 可用时才能得到 p50/p90。输入为空时
    返回 0.0，便于 smoke test 在极端参数下仍能生成完整 JSON schema。
    """

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure_latency_ms(
    run: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
    iters: int,
) -> tuple[torch.Tensor, list[float], str]:
    """测量 benchmark latency。

    返回：
    - out: 最后一次 run 的输出，用于 checksum。
    - raw_latency_ms: 每个 repeat 的单 iter 平均耗时，单位毫秒。
    - timing_method: "cuda_event" 或 "perf_counter"。

    CUDA 路径使用 CUDA event，因为 kernel launch 是异步的，event 能记录 GPU
    stream 上真正经过的时间。CPU 路径使用 perf_counter。
    """

    out = run()
    for _ in range(max(0, warmup)):
        out = run()
    synchronize(device)

    raw_latency_ms: list[float] = []
    repeats = max(1, repeats)
    iters = max(1, iters)
    if device.type == "cuda":
        timing_method = "cuda_event"
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _iter in range(iters):
                out = run()
            end.record()
            end.synchronize()
            raw_latency_ms.append(float(start.elapsed_time(end) / iters))
    else:
        timing_method = "perf_counter"
        for _ in range(repeats):
            start_time = perf_counter()
            for _iter in range(iters):
                out = run()
            elapsed = perf_counter() - start_time
            raw_latency_ms.append(float(elapsed * 1000.0 / iters))
    synchronize(device)
    return out, raw_latency_ms, timing_method


def _nsight_compute_command(args: argparse.Namespace) -> str:
    """生成可复制的 Nsight Compute profiling 命令模板。

    这个命令不会在 benchmark 中自动执行，因为 ncu 通常需要交互权限、较长时间
    和 CUDA 环境。JSON 中记录模板，保证 profiling 可复现。
    """

    output = f"results/ncu_{args.backend}_b{args.batch}_h{args.heads}_s{args.seq_len}_d{args.head_dim}"
    # 只采集回填 measured_* 需要的 metric，不用 `--set full`：full 会把每个 kernel
    # 重放几十次，在 Colab 上极慢甚至超时。`-s`/`-c` 跳过 warmup、只 profile 少量
    # launch，进一步压缩开销。
    return (
        f"ncu --metrics {NCU_METRICS} "
        f"--launch-skip {max(1, args.warmup)} --launch-count {NCU_LAUNCH_COUNT} "
        f"--target-processes all --export {output} --force-overwrite --csv "
        f"python benchmarks/microbench.py --backend {args.backend} --batch {args.batch} --heads {args.heads} "
        f"--seq-len {args.seq_len} --head-dim {args.head_dim} --block-size {args.block_size} "
        f"--iters {max(1, args.iters)} --warmup {max(1, args.warmup)} --repeats 1 "
        f"--device cuda --dtype {args.dtype} --json"
    )


def _run_ncu_and_backfill(args: argparse.Namespace, result: dict) -> dict:
    """在 Colab/CUDA 上用 ncu profile 本 backend，把 measured_* 字段回填进 result。

    做法：以子进程方式对 microbench 自身再跑一次（去掉 --profile-ncu 防止递归），
    让 ncu 只 profile warmup 之后的少量 launch，用 --csv 直接拿到指标文本后解析。
    任何失败都不会中断主流程，只在 result 里记录 profiler_error。
    """

    import subprocess

    metrics_str = NCU_METRICS
    cmd = [
        args.ncu_bin,
        "--metrics", metrics_str,
        "--launch-skip", str(max(1, args.warmup)),
        "--launch-count", str(args.ncu_launch_count),
        "--target-processes", "all",
        "--csv",
        sys.executable, str(Path(__file__).resolve()),
        "--backend", args.backend,
        "--batch", str(args.batch), "--heads", str(args.heads),
        "--seq-len", str(args.seq_len), "--head-dim", str(args.head_dim),
        "--block-size", str(args.block_size),
        "--iters", str(max(1, args.iters)), "--warmup", str(max(1, args.warmup)),
        "--repeats", "1", "--device", "cuda", "--dtype", args.dtype,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.ncu_timeout)
    except FileNotFoundError:
        result["profiler_error"] = f"找不到 ncu 可执行文件：{args.ncu_bin}（Colab 上通常在 /usr/local/cuda/bin/ncu，或 apt-get install nsight-compute）"
        return result
    except subprocess.TimeoutExpired:
        result["profiler_error"] = f"ncu profiling 超时（>{args.ncu_timeout}s），可减小 --iters 或 --ncu-launch-count"
        return result

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        hint = "\n".join(tail)
        if "ERR_NVGPUCTRPERM" in (proc.stderr or ""):
            hint += "\n[权限问题] Colab 通常以 root 运行可直接 profile；本地需 sudo 或放开 NVreg_RestrictProfilingToAdminUsers。"
        result["profiler_error"] = f"ncu 退出码 {proc.returncode}:\n{hint}"
        return result

    try:
        parsed = parse_ncu_csv(proc.stdout)
    except ValueError as exc:
        result["profiler_error"] = f"解析 ncu CSV 失败：{exc}"
        return result

    result.update(parsed.as_backfill())
    result["bandwidth_fields_are_estimates"] = False
    return result


def _latency_breakdown(
    args: argparse.Namespace,
    stats: dict[str, float],
    measured_latency_ms: float,
    timing_method: str,
) -> list[dict[str, float | str]]:
    """输出 decode attention 的阶段化 breakdown 元数据。

    当前 JSON 中的 breakdown 是“可 profile 的阶段定义 + 粗粒度工作量估算”，不是
    Nsight 实测拆分。真正的每阶段 latency 需要用 JSON 中的 ncu 命令采集 source
    line / instruction / memory metrics 后归因。
    """

    batch = float(args.batch)
    heads = float(args.heads)
    seq_len = float(args.seq_len)
    head_dim = float(args.head_dim)
    elements = batch * heads * seq_len * head_dim
    output_bytes = batch * heads * head_dim * 4.0
    quant_bytes = float(stats["quant_kv_bytes"])
    dense_bytes = float(stats["dense_kv_bytes"])
    portable_quant_backend = args.backend in {"fused", "paged"}
    triton_quant_backend = args.backend in {"triton_fused", "triton_paged"}
    kv_load_bytes = quant_bytes if triton_quant_backend else dense_bytes
    if portable_quant_backend:
        # portable quant 路径会先读取 INT8 KV 反量化，再把 dense KV 交给 reference attention；
        # 因此这里把量化 KV 读取和后续 dense KV 读取都计入阶段估算。
        kv_load_bytes = quant_bytes + dense_bytes
    if portable_quant_backend:
        kv_load_notes = "读取 K/V；portable quant backend 包含 INT8 KV 读取和 materialized dense KV 的后续读取。"
    elif triton_quant_backend:
        kv_load_notes = "读取 K/V；Triton quant backend 应主要读取 INT8 values 和 scale/zero_point。"
    else:
        kv_load_notes = "读取 dense K/V reference tensor。"
    qk_flops = 2.0 * elements
    pv_flops = 2.0 * elements
    softmax_ops = 3.0 * batch * heads * seq_len
    dequant_ops = 4.0 * elements if args.backend in {"fused", "triton_fused", "paged", "triton_paged"} else 0.0
    dequant_materialization_bytes = dense_bytes if portable_quant_backend else 0.0
    total_estimated_bytes = kv_load_bytes + dequant_materialization_bytes + output_bytes
    return [
        {
            "stage": "kv_load",
            "estimated_bytes": kv_load_bytes,
            "estimated_flops_or_ops": 0.0,
            "measurement": "ncu dram/source-line metrics",
            "notes": kv_load_notes,
        },
        {
            "stage": "dequant",
            "estimated_bytes": dequant_materialization_bytes,
            "estimated_flops_or_ops": dequant_ops,
            "measurement": "ncu source-line/instruction metrics",
            "notes": "INT8 -> float 的 zero_point/scale 反量化；portable quant backend 会额外写出 dense KV，Triton backend 不物化 dense KV。",
        },
        {
            "stage": "qk",
            "estimated_bytes": 0.0,
            "estimated_flops_or_ops": qk_flops,
            "measurement": "ncu source-line/instruction metrics",
            "notes": "QK 点积，输出每个历史 token 的 score。",
        },
        {
            "stage": "softmax",
            "estimated_bytes": 0.0,
            "estimated_flops_or_ops": softmax_ops,
            "measurement": "ncu source-line/instruction metrics",
            "notes": "online softmax 的 max、exp、sum 更新。",
        },
        {
            "stage": "pv_accumulation",
            "estimated_bytes": 0.0,
            "estimated_flops_or_ops": pv_flops,
            "measurement": "ncu source-line/instruction metrics",
            "notes": "softmax 权重乘 V 并累积输出。",
        },
        {
            "stage": "output_write",
            "estimated_bytes": output_bytes,
            "estimated_flops_or_ops": 0.0,
            "measurement": "ncu dram/source-line metrics",
            "notes": "写出 [batch, heads, head_dim] attention 结果。",
        },
        {
            "stage": "total_kernel_or_backend",
            "estimated_bytes": total_estimated_bytes,
            "estimated_flops_or_ops": qk_flops + pv_flops + softmax_ops + dequant_ops,
            "measurement": timing_method,
            "measured_latency_ms": measured_latency_ms,
            "notes": "总耗时来自 benchmark 测量；阶段 latency 需要 Nsight Compute 进一步归因。",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashSpec decode attention microbenchmark")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backend", choices=["dense", "fused", "triton_fused", "paged", "triton_paged"], default="paged")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--profile-ncu", action="store_true",
                        help="用 Nsight Compute 采集实测 DRAM/带宽/占用率并回填 measured_* 字段（需 CUDA + ncu）")
    parser.add_argument("--ncu-bin", default="ncu", help="ncu 可执行文件路径（Colab: /usr/local/cuda/bin/ncu）")
    parser.add_argument("--ncu-launch-count", type=int, default=NCU_LAUNCH_COUNT,
                        help="ncu 只 profile warmup 之后的这么多次 launch")
    parser.add_argument("--ncu-timeout", type=float, default=600.0, help="ncu 子进程超时秒数")
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

    out, raw_latency_ms, timing_method = _measure_latency_ms(
        run=run,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        iters=args.iters,
    )
    latency_ms = statistics.median(raw_latency_ms)
    latency_mean_ms = statistics.fmean(raw_latency_ms)
    latency_std_ms = statistics.pstdev(raw_latency_ms) if len(raw_latency_ms) > 1 else 0.0
    latency_min_ms = min(raw_latency_ms)
    latency_max_ms = max(raw_latency_ms)
    latency_p50_ms = _percentile(raw_latency_ms, 50.0)
    latency_p90_ms = _percentile(raw_latency_ms, 90.0)
    latency_p99_ms = _percentile(raw_latency_ms, 99.0)
    dense_bandwidth = stats["dense_kv_bytes"] / max(1.0e-9, latency_ms / 1000.0) / 1.0e9
    quant_bandwidth = stats["quant_kv_bytes"] / max(1.0e-9, latency_ms / 1000.0) / 1.0e9
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
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_method": timing_method,
        "latency_ms": latency_ms,
        "latency_mean_ms": latency_mean_ms,
        "latency_std_ms": latency_std_ms,
        "latency_min_ms": latency_min_ms,
        "latency_max_ms": latency_max_ms,
        "latency_p50_ms": latency_p50_ms,
        "latency_p90_ms": latency_p90_ms,
        "latency_p99_ms": latency_p99_ms,
        "tokens_per_second": args.batch / max(1.0e-9, latency_ms / 1000.0),
        "estimated_dense_kv_bytes": stats["dense_kv_bytes"],
        "estimated_quant_kv_bytes": stats["quant_kv_bytes"],
        "compression_ratio": stats["compression_ratio"],
        "materializes_dense_kv": bool(stats["materializes_dense_kv"]),
        "bandwidth_fields_are_estimates": True,
        "estimated_effective_dense_kv_bandwidth_gbps": dense_bandwidth,
        "estimated_effective_quant_kv_bandwidth_gbps": quant_bandwidth,
        "effective_dense_kv_bandwidth_gbps": dense_bandwidth,
        "effective_quant_kv_bandwidth_gbps": quant_bandwidth,
        "measured_kernel_latency_ms": latency_ms if timing_method == "cuda_event" else None,
        "measured_latency_p50_ms": latency_p50_ms,
        "measured_latency_p90_ms": latency_p90_ms,
        "measured_latency_p99_ms": latency_p99_ms,
        "measured_dram_bytes": None,
        "measured_achieved_bandwidth_gbps": None,
        "measured_occupancy_pct": None,
        "measured_sm_utilization_pct": None,
        "profiler_metrics_source": "nsight_compute_required_for_dram_occupancy_sm",
        "nsight_compute_command": _nsight_compute_command(args),
        "latency_breakdown": _latency_breakdown(args, stats, latency_ms, timing_method),
        "output_checksum": float(out.float().sum().item()),
    }
    if args.include_raw:
        result["raw_latency_ms"] = raw_latency_ms

    if args.profile_ncu:
        if device.type != "cuda":
            result["profiler_error"] = "--profile-ncu 需要 CUDA 设备"
        else:
            result = _run_ncu_and_backfill(args, result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(json.dumps({"raw_latency_ms": raw_latency_ms}, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
