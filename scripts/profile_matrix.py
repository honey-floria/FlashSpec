"""运行 FlashSpec kernel profiling 矩阵，并把每个点的 JSON 结果整理成 manifest。

这个脚本是矩阵 profiling 的主入口：它只负责枚举参数组合、设置环境变量、
调用 `benchmarks/microbench.py`，再把结果写成统一的 CSV，方便后续分析脚本和
Markdown 报告复用。它不直接解释性能好坏，只负责把数据跑全、跑稳、跑可追溯。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cli_common import parse_int_list as _ints, parse_str_list as _strings  # noqa: E402


def parse_args() -> argparse.Namespace:
    # 所有 matrix 参数都在这里集中声明，默认值尽量贴近当前文档里的标准实验配置。
    parser = argparse.ArgumentParser(description="Run FlashSpec kernel profiling matrices")
    parser.add_argument("--backend", choices=["triton_fused", "triton_paged"], required=True)
    parser.add_argument("--seq-lens", default="512,2048,4096")
    parser.add_argument("--head-dims", default="64,128")
    parser.add_argument("--block-ns", default="32,64,128")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--num-splits", default="auto,1,2,4,8",
                        help="Kernel 1 only. 'auto' leaves FLASHSPEC_NUM_SPLITS unset.")
    parser.add_argument("--length-patterns", default="uniform,descending",
                        help="microbench --length-pattern values")
    parser.add_argument("--paged-layouts", default="contiguous,shuffled",
                        help="Kernel 2 only. microbench --paged-layout values")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-seed", type=int, default=0)
    parser.add_argument("--profile-ncu", action="store_true",
                        help="Run microbench --profile-ncu for every matrix point")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--ncu-launch-count", type=int, default=5)
    parser.add_argument("--ncu-timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/profile_matrix"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()


def _variant_iter(args: argparse.Namespace):
    # 用 product 把每个维度的候选值展开成完整矩阵。
    # fused 和 paged 的参数空间不同，所以这里按 backend 分两条展开逻辑。
    seq_lens = _ints(args.seq_lens)
    head_dims = _ints(args.head_dims)
    block_ns = _ints(args.block_ns)
    num_warps = _ints(args.num_warps)
    length_patterns = _strings(args.length_patterns)
    if args.backend == "triton_fused":
        split_values = _strings(args.num_splits)
        for seq_len, head_dim, block_n, warps, splits, length_pattern in product(
            seq_lens, head_dims, block_ns, num_warps, split_values, length_patterns
        ):
            yield {
                "seq_len": seq_len,
                "head_dim": head_dim,
                "block_n": block_n,
                "num_warps": warps,
                "num_splits": splits,
                "length_pattern": length_pattern,
                "paged_layout": "contiguous",
            }
    else:
        paged_layouts = _strings(args.paged_layouts)
        for seq_len, head_dim, block_n, warps, length_pattern, paged_layout in product(
            seq_lens, head_dims, block_ns, num_warps, length_patterns, paged_layouts
        ):
            yield {
                "seq_len": seq_len,
                "head_dim": head_dim,
                "block_n": block_n,
                "num_warps": warps,
                "num_splits": "unused",
                "length_pattern": length_pattern,
                "paged_layout": paged_layout,
            }


def _file_name(args: argparse.Namespace, v: dict[str, object]) -> str:
    # 结果文件名把关键 knob 编进文件名，方便在 results 目录里直接按名字识别配置。
    parts = [
        args.backend,
        f"s{v['seq_len']}",
        f"d{v['head_dim']}",
        f"bn{v['block_n']}",
        f"nw{v['num_warps']}",
        f"len{v['length_pattern']}",
    ]
    if args.backend == "triton_fused":
        parts.append(f"split{v['num_splits']}")
    else:
        parts.append(f"layout{v['paged_layout']}")
    return "_".join(str(p) for p in parts) + ".json"


def _command(args: argparse.Namespace, v: dict[str, object], output: Path) -> list[str]:
    # 这里拼的是 microbench 命令，而不是直接在本脚本里重写 benchmark 逻辑。
    # 这样 profile_matrix 只是 orchestration 层，单点逻辑仍由 microbench 维护。
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "microbench.py"),
        "--backend", args.backend,
        "--batch", str(args.batch),
        "--heads", str(args.heads),
        "--seq-len", str(v["seq_len"]),
        "--head-dim", str(v["head_dim"]),
        "--block-size", str(args.block_size),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--device", args.device,
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--length-pattern", str(v["length_pattern"]),
        "--paged-layout", str(v["paged_layout"]),
        "--layout-seed", str(args.layout_seed),
        "--json",
        "--output", str(output),
    ]
    if args.profile_ncu:
        cmd += [
            "--profile-ncu",
            "--ncu-bin", args.ncu_bin,
            "--ncu-launch-count", str(args.ncu_launch_count),
            "--ncu-timeout", str(args.ncu_timeout),
        ]
    return cmd


def _env(args: argparse.Namespace, v: dict[str, object]) -> dict[str, str]:
    # matrix 的关键 knob 通过环境变量传给 Triton kernel，避免把它们硬编码在代码里。
    env = os.environ.copy()
    env["FLASHSPEC_BLOCK_N"] = str(v["block_n"])
    env["FLASHSPEC_NUM_WARPS"] = str(v["num_warps"])
    if args.backend == "triton_fused" and v["num_splits"] != "auto":
        env["FLASHSPEC_NUM_SPLITS"] = str(v["num_splits"])
    else:
        env.pop("FLASHSPEC_NUM_SPLITS", None)
    return env


def _manifest_row(path: Path, v: dict[str, object], data: dict[str, object]) -> dict[str, object]:
    # manifest 的职责是把“矩阵参数”和“实测/回填结果”放在一张表里。
    # 后续 analyze_matrix.py 只需要读这个 CSV，就能生成报告。
    return {
        "file": str(path),
        "backend": data.get("backend"),
        "seq_len": data.get("seq_len"),
        "head_dim": data.get("head_dim"),
        "block_n": data.get("block_n"),
        "num_warps": data.get("num_warps"),
        "num_splits": data.get("num_splits"),
        "length_pattern": data.get("length_pattern"),
        "effective_min_seq_len": data.get("effective_min_seq_len"),
        "effective_max_seq_len": data.get("effective_max_seq_len"),
        "paged_layout": data.get("paged_layout"),
        "latency_ms": data.get("latency_ms"),
        "measured_achieved_bandwidth_gbps": data.get("measured_achieved_bandwidth_gbps"),
        "measured_dram_throughput_pct": data.get("measured_dram_throughput_pct"),
        "measured_occupancy_pct": data.get("measured_occupancy_pct"),
        "measured_registers_per_thread": data.get("measured_registers_per_thread"),
        "measured_theoretical_occupancy_pct": data.get("measured_theoretical_occupancy_pct"),
        "profiler_error": data.get("profiler_error", ""),
        "matrix_block_n": v["block_n"],
        "matrix_num_warps": v["num_warps"],
        "matrix_num_splits": v["num_splits"],
    }


def main() -> None:
    # 主流程很直：解析参数 -> 枚举矩阵 -> 跑每个点 -> 回填 JSON -> 写 manifest。
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.summary or (args.output_dir / f"{args.backend}_manifest.csv")
    rows: list[dict[str, object]] = []

    for variant in _variant_iter(args):
        output = args.output_dir / _file_name(args, variant)
        cmd = _command(args, variant, output)
        env = _env(args, variant)
        # 把关键 knob 展示出来，日志里一眼就能看到当前点实际跑的是哪组环境变量。
        env_prefix = " ".join(
            f"{k}={env[k]}" for k in ("FLASHSPEC_BLOCK_N", "FLASHSPEC_NUM_WARPS", "FLASHSPEC_NUM_SPLITS") if k in env
        )
        print(f">> {env_prefix} {shlex.join(cmd)}")
        if args.dry_run:
            continue
        # 真实执行时把 stdout/stderr 收起来，只把失败信息显式抛给调用者。
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"matrix point failed with exit code {proc.returncode}\n{proc.stderr}")
        data = json.loads(output.read_text(encoding="utf-8"))
        rows.append(_manifest_row(output, variant, data))

    if rows:
        # 只有真正跑出结果时才写 summary CSV，避免空目录留下误导性的空文件。
        summary.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(summary)


if __name__ == "__main__":
    main()
