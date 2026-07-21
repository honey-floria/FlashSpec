from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
# 同时加入 ROOT 和 ROOT/src：flashspec 包在 src 下，scripts.ncu_parse 在 ROOT 下。
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

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
from flashspec.triton_utils import HAS_TRITON
from scripts.cli_common import DTYPE_CHOICES, emit_result, microbench_cli_args
from scripts.ncu_parse import apply_backfill, parse_ncu_csv

# 回填 measured_* 字段所需的最小 ncu metric 集合（与 ncu_parse 对齐）。
NCU_METRICS = (
    "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    # 占用率限制诊断：每线程寄存器数 + 理论占用率上限（越低说明越受资源限制）。
    # 用于判定 occupancy 卡在 ~25% 是否因寄存器压力（而非 block 数不足）。
    "launch__registers_per_thread,"
    "sm__maximum_warps_per_active_cycle_pct"
)
NCU_SOURCE_SECTIONS = ("SourceCounters", "InstructionStats", "MemoryWorkloadAnalysis", "SchedulerStats")
NCU_LAUNCH_COUNT = 5  # 默认只 profile 这么多次 launch（匹配到的），控制 ncu 开销。

# 各 backend 真正要 profile 的 kernel 名（正则）。用名字过滤而非 launch 序号，
# 才能避开 backend 在计时前做的量化 elementwise kernel（Div/round/clamp/add），
# 否则 ncu 会把稠密 KV 的量化访存当成 attention 访存，实测字节严重偏大。
_NCU_KERNEL_REGEX = {
    # Split-K 后 fused 有 3 个 kernel：单 kernel(_fused_dequant_attention_kernel)、
    # split kernel(_fused_dequant_attention_split_kernel) 和 combine(_combine_splits_kernel)。
    # 正则要覆盖全部三个，否则 Split-K 开启时 ncu 抓不到 kernel。
    "triton_fused": "fused_dequant_attention|combine_splits",
    "triton_paged": "paged_quant_attention_kernel|combine_splits",
    "dense": "attention",  # reference_attention 的 SDPA/矩阵 kernel
}


def _ncu_kernel_regex(args: argparse.Namespace) -> str | None:
    """返回该 backend 应 profile 的 kernel 名正则；未知则不过滤。"""
    if getattr(args, "ncu_kernel_regex", ""):
        return args.ncu_kernel_regex
    return _NCU_KERNEL_REGEX.get(args.backend)


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


def _comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_lengths(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """生成本次 benchmark 的每 request 有效长度。

    默认全是 seq_len。profiling Kernel 2 时可用 --length-pattern 或 --lengths
    生成 variable length case，观察 mask 分支和 block_table 扫描范围的影响。
    """

    if args.lengths:
        values = _comma_ints(args.lengths)
        if len(values) == 1:
            values = values * args.batch
        if len(values) != args.batch:
            raise ValueError("--lengths 必须给 1 个值或 batch 个逗号分隔值")
    else:
        pattern = args.length_pattern
        if pattern == "uniform":
            values = [args.seq_len] * args.batch
        elif pattern == "descending":
            lo = max(1, args.seq_len // 4)
            if args.batch == 1:
                values = [args.seq_len]
            else:
                values = [
                    int(round(args.seq_len - (args.seq_len - lo) * i / (args.batch - 1)))
                    for i in range(args.batch)
                ]
        elif pattern == "bimodal":
            short = max(1, args.seq_len // 4)
            split = args.batch // 2
            values = [args.seq_len] * split + [short] * (args.batch - split)
        elif pattern == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(args.seed))
            lo = max(1, args.seq_len // 4)
            values = torch.randint(lo, args.seq_len + 1, (args.batch,), generator=generator).tolist()
        else:
            raise ValueError("--length-pattern 必须是 uniform、descending、bimodal 或 random")

    if any(v < 1 or v > args.seq_len for v in values):
        raise ValueError("effective lengths 必须位于 [1, seq_len]")
    return torch.tensor(values, dtype=torch.int64, device=device)


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


def _microbench_cli_args(args: argparse.Namespace, *, include_json: bool) -> list[str]:
    """复现当前 benchmark 配置的 microbench CLI 参数（ncu profiling 固定单 repeat + CUDA）。

    include_json 控制是否带 --json：可复制的 profiling 命令模板需要 JSON 输出，
    而 ncu 回填子进程刻意不带 --json，避免 JSON blob 混进 ncu 的 --csv stdout。
    """

    return microbench_cli_args(
        backend=args.backend,
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        block_size=args.block_size,
        iters=args.iters,
        warmup=args.warmup,
        repeats=1,
        device="cuda",
        dtype=args.dtype,
        paged_layout=args.paged_layout,
        layout_seed=args.layout_seed,
        length_pattern=args.length_pattern,
        seed=args.seed,
        lengths=args.lengths,
        include_json_flag=include_json,
    )


def _microbench_command_args(args: argparse.Namespace) -> list[str]:
    """返回可复制的 profiling 命令（含前导 python + 脚本路径 + --json）。"""

    return ["python", "benchmarks/microbench.py", *_microbench_cli_args(args, include_json=True)]


def _env_command_prefix() -> str:
    parts = []
    for key in ("FLASHSPEC_NUM_SPLITS", "FLASHSPEC_BLOCK_N", "FLASHSPEC_NUM_WARPS"):
        value = os.environ.get(key)
        if value:
            parts.append(f"{key}={value}")
    return (" ".join(parts) + " ") if parts else ""


def _nsight_compute_command(args: argparse.Namespace) -> str:
    """生成可复制的 Nsight Compute profiling 命令模板。

    这个命令不会在 benchmark 中自动执行，因为 ncu 通常需要交互权限、较长时间
    和 CUDA 环境。JSON 中记录模板，保证 profiling 可复现。
    """

    output = f"results/ncu_{args.backend}_b{args.batch}_h{args.heads}_s{args.seq_len}_d{args.head_dim}"
    # 只采集回填 measured_* 需要的 metric，不用 `--set full`：full 会把每个 kernel
    # 重放几十次，在 Colab 上极慢甚至超时。用 --kernel-name 只 profile attention
    # kernel（避开量化 elementwise kernel），--launch-count 再限制匹配次数。
    regex = _ncu_kernel_regex(args)
    kernel_filter = f'--kernel-name regex:"{regex}" ' if regex else ""
    bench = " ".join(_microbench_command_args(args))
    return (
        f"{_env_command_prefix()}ncu --metrics {NCU_METRICS} "
        f"{kernel_filter}--launch-count {NCU_LAUNCH_COUNT} "
        f"--target-processes all --export {output} --force-overwrite --csv "
        f"{bench}"
    )


def _nsight_compute_source_command(args: argparse.Namespace) -> str:
    """生成 source-line / instruction 归因用的 Nsight Compute 命令模板。"""

    output = f"results/ncu_source_{args.backend}_b{args.batch}_h{args.heads}_s{args.seq_len}_d{args.head_dim}"
    regex = _ncu_kernel_regex(args)
    kernel_filter = f'--kernel-name regex:"{regex}" ' if regex else ""
    sections = " ".join(f"--section {section}" for section in NCU_SOURCE_SECTIONS)
    bench = " ".join(_microbench_command_args(args))
    return (
        f"{_env_command_prefix()}ncu {sections} {kernel_filter}--launch-count 1 "
        f"--target-processes all --export {output} --force-overwrite "
        f"{bench}"
    )


def _run_ncu_and_backfill(args: argparse.Namespace, result: dict) -> dict:
    """在 Colab/CUDA 上用 ncu profile 本 backend，把 measured_* 字段回填进 result。

    做法：以子进程方式对 microbench 自身再跑一次（去掉 --profile-ncu 防止递归），
    让 ncu 只 profile warmup 之后的少量 launch，用 --csv 直接拿到指标文本后解析。
    任何失败都不会中断主流程，只在 result 里记录 profiler_error。
    """

    import subprocess

    cmd = [args.ncu_bin, "--metrics", NCU_METRICS]
    regex = _ncu_kernel_regex(args)
    if regex:
        cmd += ["--kernel-name", f"regex:{regex}"]
    cmd += [
        "--launch-count", str(args.ncu_launch_count),
        "--target-processes", "all",
        "--csv",
        # 用当前解释器和脚本绝对路径复现同一份 benchmark 配置（子进程去掉 --profile-ncu 防递归）。
        # 不带 --json：避免 microbench 的 JSON 输出混入 ncu 的 --csv stdout 干扰解析。
        sys.executable, str(Path(__file__).resolve()),
        *_microbench_cli_args(args, include_json=False),
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

    apply_backfill(result, parsed)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", default="",
                        help="逗号分隔的每 request 有效长度；给 1 个值时广播到 batch")
    parser.add_argument("--length-pattern", default="uniform",
                        choices=["uniform", "descending", "bimodal", "random"],
                        help="未显式给 --lengths 时生成 effective lengths 的模式")
    parser.add_argument("--paged-layout", default="contiguous",
                        choices=["contiguous", "shuffled", "interleaved"],
                        help="PagedKVCache physical block 布局，用于 Kernel 2 locality profiling")
    parser.add_argument("--layout-seed", type=int, default=0, help="shuffled paged layout 的随机种子")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backend", choices=["dense", "fused", "triton_fused", "paged", "triton_paged"], default="paged")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=DTYPE_CHOICES)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--profile-ncu", action="store_true",
                        help="用 Nsight Compute 采集实测 DRAM/带宽/占用率并回填 measured_* 字段（需 CUDA + ncu）")
    parser.add_argument("--ncu-bin", default="ncu", help="ncu 可执行文件路径（Colab: /usr/local/cuda/bin/ncu）")
    parser.add_argument("--ncu-launch-count", type=int, default=NCU_LAUNCH_COUNT,
                        help="ncu 只 profile 匹配到的这么多次 launch")
    parser.add_argument("--ncu-kernel-regex", default="",
                        help="覆盖按 backend 推断的 kernel 名正则；只 profile 匹配的 kernel")
    parser.add_argument("--ncu-timeout", type=float, default=600.0, help="ncu 子进程超时秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    q = torch.randn((args.batch, args.heads, args.head_dim), generator=generator, device=device, dtype=dtype)
    k = torch.randn((args.batch, args.heads, args.seq_len, args.head_dim), generator=generator, device=device, dtype=dtype)
    v = torch.randn((args.batch, args.heads, args.seq_len, args.head_dim), generator=generator, device=device, dtype=dtype)
    lengths = _resolve_lengths(args, device)
    # 默认 uniform case 不向 Kernel 1/dense 传 lengths，保持与原始固定长度
    # benchmark 的编译路径一致；只有显式 --lengths 或 variable pattern 才测试 mask 分支。
    attention_lengths = None if args.length_pattern == "uniform" and not args.lengths else lengths

    if args.backend == "dense":
        def run() -> torch.Tensor:
            return reference_attention(q, k, v, lengths=attention_lengths)

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
            return fused_dequant_attention(q, kq, vq, lengths=attention_lengths)

        _, stats = fused_dequant_attention(q, kq, vq, lengths=attention_lengths, return_stats=True)
    elif args.backend == "triton_fused":
        if not HAS_TRITON:
            raise RuntimeError("triton_fused backend 需要安装 Triton：python -m pip install -e \".[triton]\"")
        if device.type != "cuda":
            raise RuntimeError("triton_fused backend 需要 CUDA 设备，请使用 --device cuda 或 --device auto")

        kq = quantize_int8_per_block(k, block_size=args.block_size)
        vq = quantize_int8_per_block(v, block_size=args.block_size)

        def run() -> torch.Tensor:
            return fused_dequant_attention_triton(q, kq, vq, lengths=attention_lengths)

        _, stats = fused_dequant_attention_triton(q, kq, vq, lengths=attention_lengths, return_stats=True)
    elif args.backend == "paged":
        cache = PagedKVCache.from_dense(
            k,
            v,
            block_size=args.block_size,
            lengths=attention_lengths,
            block_table_pattern=args.paged_layout,
            layout_seed=args.layout_seed,
        )

        def run() -> torch.Tensor:
            return paged_quant_attention(q, cache)

        _, stats = paged_quant_attention(q, cache, return_stats=True)
    else:
        if not HAS_TRITON:
            raise RuntimeError("triton_paged backend 需要安装 Triton：python -m pip install -e \".[triton]\"")
        if device.type != "cuda":
            raise RuntimeError("triton_paged backend 需要 CUDA 设备，请使用 --device cuda 或 --device auto")

        cache = PagedKVCache.from_dense(
            k,
            v,
            block_size=args.block_size,
            lengths=attention_lengths,
            block_table_pattern=args.paged_layout,
            layout_seed=args.layout_seed,
        )

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
        "seed": args.seed,
        "length_pattern": args.length_pattern,
        "passes_lengths_to_attention": attention_lengths is not None,
        "effective_lengths": [int(x) for x in lengths.detach().cpu().tolist()],
        "effective_min_seq_len": int(lengths.min().item()),
        "effective_max_seq_len": int(lengths.max().item()),
        "paged_layout": args.paged_layout,
        "paged_layout_seed": args.layout_seed,
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
        # Split-K 段数：仅 triton_fused 会给出。None=该 backend 无此概念，
        # 1=走单 kernel 快路径，>1=走 split+combine 路径。用于验证 Split-K 是否生效。
        "num_splits": stats.get("num_splits"),
        # kernel tile / 环境覆盖：后续优化实验必须能从 JSON 反查真实运行配置。
        # block_n 由 Triton backend stats 回填；portable backend 无此概念，为 None。
        "block_n": stats.get("block_n"),
        "num_warps": stats.get("num_warps"),
        "env_flashspec_num_splits": os.environ.get("FLASHSPEC_NUM_SPLITS"),
        "env_flashspec_block_n": os.environ.get("FLASHSPEC_BLOCK_N"),
        "env_flashspec_num_warps": os.environ.get("FLASHSPEC_NUM_WARPS"),
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
        "measured_dram_throughput_pct": None,
        "measured_registers_per_thread": None,
        "measured_theoretical_occupancy_pct": None,
        "measured_ncu_kernel_duration_ms": None,
        "measured_ncu_kernel_count": None,
        "measured_ncu_kernel_names": None,
        "profiler_metrics_source": "nsight_compute_required_for_dram_occupancy_sm",
        "nsight_compute_command": _nsight_compute_command(args),
        "nsight_compute_source_command": _nsight_compute_source_command(args),
        "nsight_compute_commands": {
            "metrics_csv": _nsight_compute_command(args),
            "source_instruction_report": _nsight_compute_source_command(args),
        },
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
    emit_result(result, as_json=args.json)


if __name__ == "__main__":
    main()
