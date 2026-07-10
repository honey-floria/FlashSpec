from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析 profiling report 命令行参数。"""

    parser = argparse.ArgumentParser(description="Summarize FlashSpec benchmark profiling fields")
    parser.add_argument("input", type=Path, help="microbench JSON output")
    parser.add_argument("--output", type=Path, default=Path("results/profile_report.md"))
    return parser.parse_args()


def _format_metric(value: object) -> str:
    """把 JSON 字段格式化成 Markdown 中稳定可读的字符串。

    `None` 表示该 measured profiler 字段还没有通过 Nsight Compute 回填；
    float 保留 6 位有效数字，避免报告里出现过长的小数。
    """

    if value is None:
        return "not_collected"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    """把 microbench JSON 转成 Markdown profiling report。

    microbench 负责产生 CUDA event latency、估算 bandwidth、Nsight Compute
    命令模板和 latency_breakdown；本脚本只做报告整理，不会自动运行 ncu。
    """

    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    breakdown = data.get("latency_breakdown", [])
    raw_latency = data.get("raw_latency_ms", [])
    lines = [
        "# FlashSpec Profiling Report",
        "",
        f"- backend: `{data.get('backend')}`",
        f"- device: `{data.get('device_name')}`",
        f"- shape: batch={data.get('batch')}, heads={data.get('heads')}, seq_len={data.get('seq_len')}, head_dim={data.get('head_dim')}",
        f"- timing_method: `{data.get('timing_method')}`",
        f"- latency_ms: `{_format_metric(data.get('latency_ms'))}`",
        f"- latency_p50_ms: `{_format_metric(data.get('latency_p50_ms'))}`",
        f"- latency_p90_ms: `{_format_metric(data.get('latency_p90_ms'))}`",
        f"- latency_p99_ms: `{_format_metric(data.get('latency_p99_ms'))}`",
        f"- tokens_per_second: `{_format_metric(data.get('tokens_per_second'))}`",
        f"- materializes_dense_kv: `{data.get('materializes_dense_kv')}`",
        f"- raw_latency_samples: `{len(raw_latency)}`",
        "",
        "## Nsight Compute Command",
        "",
        "```bash",
        str(data.get("nsight_compute_command", "")),
        "```",
        "",
        "## Latency Breakdown Map",
        "",
        "| stage | estimated bytes | estimated flops/ops | measurement | notes |",
        "|---|---:|---:|---|---|",
    ]
    for item in breakdown:
        # breakdown 由 microbench 生成，这里只负责 Markdown 表格转义。
        lines.append(
            "| {stage} | {estimated_bytes} | {estimated_flops_or_ops} | {measurement} | {notes} |".format(
                stage=item.get("stage", ""),
                estimated_bytes=_format_metric(item.get("estimated_bytes", "")),
                estimated_flops_or_ops=_format_metric(item.get("estimated_flops_or_ops", "")),
                measurement=item.get("measurement", ""),
                notes=str(item.get("notes", "")).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Measured Profiler Fields",
            "",
            f"- measured_kernel_latency_ms: `{_format_metric(data.get('measured_kernel_latency_ms'))}`",
            f"- measured_dram_bytes: `{_format_metric(data.get('measured_dram_bytes'))}`",
            f"- measured_achieved_bandwidth_gbps: `{_format_metric(data.get('measured_achieved_bandwidth_gbps'))}`",
            f"- measured_occupancy_pct: `{_format_metric(data.get('measured_occupancy_pct'))}`",
            f"- measured_sm_utilization_pct: `{_format_metric(data.get('measured_sm_utilization_pct'))}`",
            "",
            "这些 measured profiler 字段需要用上面的 Nsight Compute 命令采集后回填；microbench 默认只负责生成可复现命令和 CUDA event latency。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
