"""把 `results/profile_matrix` 和 source attribution 导出整理成稳定的 Markdown 报告。

这个脚本不做任何 profiling，只负责把 matrix manifest、NCU 导出 CSV 和关键派生
指标汇总成一份可读报告。这样可以避免手工翻多个目录，也方便比较不同 backend、
不同 shape 和不同参数组的平均表现。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def parse_args() -> argparse.Namespace:
    # 这里的默认目录都指向仓库内的正式结果目录，便于在 Colab 或本地直接复用。
    parser = argparse.ArgumentParser(description="Generate a stable FlashSpec profiling matrix report")
    parser.add_argument("--matrix-dir", type=Path, default=Path("results/profile_matrix"))
    parser.add_argument("--source-dir", type=Path, default=Path("results/ncu_source_attribution_export"))
    parser.add_argument("--output", type=Path, default=Path("results/profile_matrix_report.md"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _float(value: object) -> float | None:
    # matrix manifest 里的字段来自 CSV，可能是空字符串、带逗号的数字或普通文本。
    # 这里统一转成 float，失败则返回 None，避免后面排序/聚合时反复写容错逻辑。
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    # 所有输入都按 UTF-8 CSV 读取；脚本不依赖 pandas，便于在最小环境下运行。
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _manifest_rows(matrix_dir: Path) -> list[dict[str, str]]:
    # 遍历所有 *_manifest.csv，把不同矩阵的行合并成一个大表。
    # 每行额外写入 `_manifest`，便于报告里追溯到具体清单文件。
    rows: list[dict[str, str]] = []
    for path in sorted(matrix_dir.glob("**/*_manifest.csv")):
        for row in _read_csv(path):
            row["_manifest"] = str(path)
            rows.append(row)
    return rows


def _group(rows: Iterable[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    # 按一组 key 聚合成 defaultdict(list)，后面既能算平均值，也能按组合输出表格。
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def _avg(rows: list[dict[str, str]], field: str) -> float | None:
    # 某些行可能缺 profiler 字段或是空值，所以先过滤 None 再取均值。
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    # 报告里统一用固定小数位，避免不同机器/不同 Python 版本产生过长浮点串。
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _top_table(rows: list[dict[str, str]], top_k: int) -> list[str]:
    # 每个 shape 先按 latency 排序，输出最优的前 K 个点，方便快速定位默认候选。
    lines = [
        "| backend | seq | head_dim | latency ms | block_n | warps | splits | len pattern | layout | BW GB/s | DRAM % | occ % | regs |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    sorted_rows = sorted(rows, key=lambda row: _float(row.get("latency_ms")) or float("inf"))
    for row in sorted_rows[:top_k]:
        lines.append(
            "| {backend} | {seq} | {head_dim} | {lat} | {bn} | {warps} | {splits} | {pattern} | {layout} | {bw} | {dram} | {occ} | {regs} |".format(
                backend=row.get("backend", ""),
                seq=row.get("seq_len", ""),
                head_dim=row.get("head_dim", ""),
                lat=_fmt(_float(row.get("latency_ms"))),
                bn=row.get("block_n", ""),
                warps=row.get("num_warps", ""),
                splits=row.get("num_splits", ""),
                pattern=row.get("length_pattern", ""),
                layout=row.get("paged_layout", ""),
                bw=_fmt(_float(row.get("measured_achieved_bandwidth_gbps")), 1),
                dram=_fmt(_float(row.get("measured_dram_throughput_pct")), 1),
                occ=_fmt(_float(row.get("measured_occupancy_pct")), 1),
                regs=_fmt(_float(row.get("measured_registers_per_thread")), 0),
            )
        )
    return lines


def _group_table(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[str]:
    # 对同一 backend 下的参数维度做分组平均，回答“某个 knob 整体上是好还是坏”。
    header = " | ".join(keys)
    lines = [
        f"| {header} | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |",
        "|" + "---|" * (len(keys) + 6),
    ]
    grouped = _group(rows, keys)
    for key, items in sorted(grouped.items()):
        lines.append(
            "| {key} | {count} | {lat} | {bw} | {dram} | {occ} | {regs} |".format(
                key=" | ".join(value or "n/a" for value in key),
                count=len(items),
                lat=_fmt(_avg(items, "latency_ms")),
                bw=_fmt(_avg(items, "measured_achieved_bandwidth_gbps"), 1),
                dram=_fmt(_avg(items, "measured_dram_throughput_pct"), 1),
                occ=_fmt(_avg(items, "measured_occupancy_pct"), 1),
                regs=_fmt(_avg(items, "measured_registers_per_thread"), 0),
            )
        )
    return lines


def _source_rows(source_dir: Path) -> list[dict[str, object]]:
    # source attribution 导出的 CSV 里每个文件对应一个 case。
    # 这里保留主 kernel，跳过 combine kernel，因为 combine 只是配套收尾，容易稀释主 kernel 的特征。
    rows: list[dict[str, object]] = []
    for path in sorted(source_dir.glob("*.csv")):
        for row in _read_csv(path):
            kernel = row.get("Kernel Name", "")
            if not kernel or "combine" in kernel:
                continue
            sample = _float(row.get("smsp__pcsamp_sample_count")) or 0.0
            rows.append(
                {
                    "case": path.stem,
                    "kernel": kernel,
                    "time_us": _float(row.get("gpu__time_duration.sum")),
                    "regs": _float(row.get("launch__registers_per_thread")),
                    "theo_occ": _float(row.get("sm__maximum_warps_per_active_cycle_pct")),
                    "dram_gbs": _float(row.get("dram__bytes.sum.per_second")),
                    "issue": _float(row.get("smsp__issue_active.avg.per_cycle_active")),
                    "active_warps": _float(row.get("smsp__warps_active.avg.per_cycle_active")),
                    "eligible_warps": _float(row.get("smsp__warps_eligible.avg.per_cycle_active")),
                    "long_pct": _pct(row, sample, "smsp__pcsamp_warps_issue_stalled_long_scoreboard"),
                    "short_pct": _pct(row, sample, "smsp__pcsamp_warps_issue_stalled_short_scoreboard"),
                    "wait_pct": _pct(row, sample, "smsp__pcsamp_warps_issue_stalled_wait"),
                    "mio_pct": _pct(row, sample, "smsp__pcsamp_warps_issue_stalled_mio_throttle"),
                    "local_sectors": _float(row.get("memory_l2_theoretical_sectors_local")),
                    "l2_global": _float(row.get("memory_l2_theoretical_sectors_global")),
                    "l2_ideal": _float(row.get("memory_l2_theoretical_sectors_global_ideal")),
                }
            )
    return rows


def _pct(row: dict[str, str], sample: float, field: str) -> float | None:
    # Nsight Compute 的某些 counters 是 sample count，需要除以总 sample 再换算成百分比。
    if sample <= 0:
        return None
    value = _float(row.get(field)) or 0.0
    return 100.0 * value / sample


def _source_table(rows: list[dict[str, object]]) -> list[str]:
    # 如果没有导出的 source attribution CSV，就给出显式提示，而不是生成空表。
    if not rows:
        return ["No source attribution CSV files found."]
    lines = [
        "| case | time us | regs | theo occ % | DRAM GB/s | issue/cycle | eligible warps | long % | short % | wait % | mio % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {time} | {regs} | {occ} | {dram} | {issue} | {eligible} | {long} | {short} | {wait} | {mio} |".format(
                case=row["case"],
                time=_fmt(row["time_us"], 1),
                regs=_fmt(row["regs"], 0),
                occ=_fmt(row["theo_occ"], 2),
                dram=_fmt(row["dram_gbs"], 1),
                issue=_fmt(row["issue"], 2),
                eligible=_fmt(row["eligible_warps"], 2),
                long=_fmt(row["long_pct"], 1),
                short=_fmt(row["short_pct"], 1),
                wait=_fmt(row["wait_pct"], 1),
                mio=_fmt(row["mio_pct"], 1),
            )
        )
    return lines


def main() -> None:
    # 1) 收集 matrix manifest
    # 2) 按 shape / backend / knob 组织表格
    # 3) 汇总 source attribution CSV
    # 4) 输出稳定 Markdown
    args = parse_args()
    rows = _manifest_rows(args.matrix_dir)
    if not rows:
        raise FileNotFoundError(f"No *_manifest.csv files found under {args.matrix_dir}")

    lines = [
        "# FlashSpec Profiling Matrix Report",
        "",
        f"- matrix_dir: `{args.matrix_dir}`",
        f"- source_dir: `{args.source_dir}`",
        f"- matrix_points: `{len(rows)}`",
        "",
        "## Top Latency Points",
    ]
    for key, items in sorted(_group(rows, ("backend", "seq_len", "head_dim")).items()):
        lines.extend(["", f"### {' / '.join(key)}", ""])
        lines.extend(_top_table(items, args.top_k))

    lines.extend(["", "## Parameter Averages", ""])
    for backend, items in sorted(_group(rows, ("backend",)).items()):
        backend_name = backend[0]
        lines.extend(["", f"### {backend_name}: block_n / num_warps", ""])
        lines.extend(_group_table(items, ("block_n", "num_warps")))
        if backend_name == "triton_fused":
            lines.extend(["", f"### {backend_name}: num_splits", ""])
            lines.extend(_group_table(items, ("num_splits",)))
        if backend_name == "triton_paged":
            lines.extend(["", f"### {backend_name}: paged_layout", ""])
            lines.extend(_group_table(items, ("paged_layout",)))

    lines.extend(["", "## Source Attribution Summary", ""])
    lines.extend(_source_table(_source_rows(args.source_dir)))
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
