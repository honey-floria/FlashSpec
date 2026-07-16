from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析 profiling report 命令行参数。"""

    # 这个脚本只接受一个 microbench JSON 输入，再把它渲染成 Markdown 报告。
    parser = argparse.ArgumentParser(description="Summarize FlashSpec benchmark profiling fields")
    parser.add_argument("input", type=Path, help="microbench JSON output")
    parser.add_argument("--output", type=Path, default=Path("results/profile_report.md"))
    return parser.parse_args()


def _format_metric(value: object) -> str:
    """把 JSON 字段格式化成 Markdown 中稳定可读的字符串。

    `None` 表示该 measured profiler 字段还没有通过 Nsight Compute 回填；
    float 保留 6 位有效数字，避免报告里出现过长的小数。
    """

    # 报告不关心 Python 原生类型，只关心输出是否稳定、是否便于人读。
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
        # 把最重要的实验上下文放在开头，方便复制到文档或 issue 中。
        "# FlashSpec Profiling Report",
        "",
        f"- backend: `{data.get('backend')}`",
        f"- device: `{data.get('device_name')}`",
        f"- shape: batch={data.get('batch')}, heads={data.get('heads')}, seq_len={data.get('seq_len')}, head_dim={data.get('head_dim')}",
        f"- kernel_knobs: block_n=`{_format_metric(data.get('block_n'))}`, num_splits=`{_format_metric(data.get('num_splits'))}`, num_warps=`{_format_metric(data.get('num_warps'))}`",
        f"- env_knobs: FLASHSPEC_BLOCK_N=`{_format_metric(data.get('env_flashspec_block_n'))}`, FLASHSPEC_NUM_SPLITS=`{_format_metric(data.get('env_flashspec_num_splits'))}`, FLASHSPEC_NUM_WARPS=`{_format_metric(data.get('env_flashspec_num_warps'))}`",
        f"- length_profile: pattern=`{data.get('length_pattern')}`, min=`{_format_metric(data.get('effective_min_seq_len'))}`, max=`{_format_metric(data.get('effective_max_seq_len'))}`",
        f"- paged_layout: `{data.get('paged_layout')}` seed=`{_format_metric(data.get('paged_layout_seed'))}`",
        f"- timing_method: `{data.get('timing_method')}`",
        f"- latency_ms: `{_format_metric(data.get('latency_ms'))}`",
        f"- latency_p50_ms: `{_format_metric(data.get('latency_p50_ms'))}`",
        f"- latency_p90_ms: `{_format_metric(data.get('latency_p90_ms'))}`",
        f"- latency_p99_ms: `{_format_metric(data.get('latency_p99_ms'))}`",
        f"- tokens_per_second: `{_format_metric(data.get('tokens_per_second'))}`",
        f"- materializes_dense_kv: `{data.get('materializes_dense_kv')}`",
        f"- raw_latency_samples: `{len(raw_latency)}`",
        "",
        "## Nsight Compute Commands",
        "",
        "Fast metrics / JSON backfill:",
        "",
        "```bash",
        str(data.get("nsight_compute_command", "")),
        "```",
        "",
        "Source-line / instruction attribution:",
        "",
        "```bash",
        str(data.get("nsight_compute_source_command", "")),
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
            # 这些字段只有在跑过 ncu 回填之后才会出现；没采集到时会显示 not_collected。
            f"- measured_kernel_latency_ms: `{_format_metric(data.get('measured_kernel_latency_ms'))}`",
            f"- measured_dram_bytes: `{_format_metric(data.get('measured_dram_bytes'))}`",
            f"- measured_achieved_bandwidth_gbps: `{_format_metric(data.get('measured_achieved_bandwidth_gbps'))}`",
            f"- measured_occupancy_pct: `{_format_metric(data.get('measured_occupancy_pct'))}`",
            f"- measured_sm_utilization_pct: `{_format_metric(data.get('measured_sm_utilization_pct'))}`",
            f"- measured_dram_throughput_pct: `{_format_metric(data.get('measured_dram_throughput_pct'))}`",
            f"- measured_registers_per_thread: `{_format_metric(data.get('measured_registers_per_thread'))}`",
            f"- measured_theoretical_occupancy_pct: `{_format_metric(data.get('measured_theoretical_occupancy_pct'))}`",
            "",
            "这些 measured profiler 字段需要用 fast metrics 命令采集后回填；source-line / instruction 命令用于进一步归因，不会自动折算成 JSON 字段。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
