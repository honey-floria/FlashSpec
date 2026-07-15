from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all(results_dir: Path) -> list[dict]:
    """读取 results 目录下全部 microbench JSON。"""
    rows = []
    for path in sorted(glob.glob(str(results_dir / "*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["_file"] = os.path.basename(path)
        rows.append(data)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FlashSpec results into plots")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    return parser.parse_args()


def plot_backend_comparison(rows: list[dict], out: Path) -> None:
    """基准 shape (s2048/d128) 下各后端延迟对比柱状图。"""
    ref = [r for r in rows if r.get("seq_len") == 2048 and r.get("head_dim") == 128
           and "_compare" not in r["_file"]]
    ref.sort(key=lambda r: r["latency_ms"])
    labels = [f"{r['backend']}\n({r['_file'].split('_s')[0]})" for r in ref]
    lat = [r["latency_ms"] for r in ref]
    std = [r.get("latency_std_ms", 0.0) for r in ref]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, lat, yerr=std, capsize=4, color="#4a7fb0")
    ax.set_ylabel("Latency (ms)  —  lower is better")
    ax.set_title("Backend latency @ batch=16 heads=32 seq=2048 head_dim=128 (A100)")
    for b, r in zip(bars, ref):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{r['latency_ms']:.3f}ms\n{r['tokens_per_second']/1000:.1f}k tok/s",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "backend_comparison_s2048_d128.png", dpi=140)
    fig.savefig(out / "backend_comparison_s2048_d128.svg")
    plt.close(fig)


def plot_triton_scaling(rows: list[dict], out: Path) -> None:
    """Triton 后端随 seq_len 的延迟与带宽 scaling。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for backend in ("triton_fused", "triton_paged"):
        for hd in (64, 128):
            pts = [r for r in rows if r.get("backend") == backend
                   and r.get("head_dim") == hd and "_compare" not in r["_file"]]
            pts.sort(key=lambda r: r["seq_len"])
            if not pts:
                continue
            seqs = [r["seq_len"] for r in pts]
            lat = [r["latency_ms"] for r in pts]
            # 有 ncu 实测带宽就优先用实测，否则回退估算，并在 label 标注。
            bw, measured_any = [], False
            for r in pts:
                m = r.get("measured_achieved_bandwidth_gbps")
                if m is not None:
                    bw.append(m)
                    measured_any = True
                else:
                    bw.append(r["effective_quant_kv_bandwidth_gbps"])
            marker = "o-" if backend == "triton_fused" else "s--"
            suffix = " [measured]" if measured_any else " [est]"
            label = f"{backend} d={hd}{suffix}"
            ax1.plot(seqs, lat, marker, label=f"{backend} d={hd}")
            ax2.plot(seqs, bw, marker, label=label)
    ax1.set(xlabel="seq_len", ylabel="Latency (ms)", title="Triton latency vs seq_len")
    ax2.set(xlabel="seq_len", ylabel="Quant KV bandwidth (GB/s)",
            title="Triton bandwidth vs seq_len (measured if ncu-backfilled, else est.)")
    ax2.axhline(1555, color="gray", ls=":", lw=1, label="A100 HBM peak ~1555")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks([512, 2048, 4096])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "triton_scaling.png", dpi=140)
    fig.savefig(out / "triton_scaling.svg")
    plt.close(fig)


def write_summary_csv(rows: list[dict], out: Path) -> None:
    """把关键指标导出成 CSV 便于二次分析。"""
    cols = [
        "_file", "backend", "seq_len", "head_dim", "block_size", "block_n", "num_warps",
        "num_splits", "env_flashspec_num_splits", "env_flashspec_block_n", "env_flashspec_num_warps",
        "length_pattern", "passes_lengths_to_attention", "effective_min_seq_len", "effective_max_seq_len",
        "paged_layout", "paged_layout_seed",
        "latency_ms", "latency_std_ms", "tokens_per_second",
        "compression_ratio", "effective_quant_kv_bandwidth_gbps",
        "measured_achieved_bandwidth_gbps", "measured_dram_throughput_pct",
        "measured_occupancy_pct", "measured_registers_per_thread",
        "measured_theoretical_occupancy_pct", "materializes_dense_kv",
    ]
    lines = [",".join(cols)]
    for r in sorted(rows, key=lambda r: (r.get("backend", ""), r.get("seq_len", 0),
                                         r.get("head_dim", 0))):
        values = []
        for c in cols:
            v = r.get(c, "")
            values.append("" if v is None else str(v))
        lines.append(",".join(values))
    (out / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_all(args.results_dir)
    plot_backend_comparison(rows, args.output_dir)
    plot_triton_scaling(rows, args.output_dir)
    write_summary_csv(rows, args.output_dir)
    print(f"wrote plots + summary.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
