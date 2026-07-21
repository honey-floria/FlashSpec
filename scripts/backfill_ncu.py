"""把 Nsight Compute 采集到的实测指标回填进 microbench 的 JSON。

用法（两种输入都支持）：

    # 1) 直接喂 ncu 导出的 CSV
    python scripts/backfill_ncu.py results/triton_fused_s2048_d128.json \\
        --ncu-csv results/ncu_triton_fused.csv

    # 2) 喂 .ncu-rep，本脚本自动调用 `ncu --import ... --csv` 转换
    python scripts/backfill_ncu.py results/triton_fused_s2048_d128.json \\
        --ncu-rep results/ncu_triton_fused_b16_h32_s2048_d128.ncu-rep

回填后会同时给出实测 vs 估算带宽的对比，便于判断 kernel 是否有多余访存。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# scripts.ncu_parse 是同目录下的兄弟模块，需要 ROOT（而非 ROOT/src）在 sys.path 上。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ncu_parse import apply_backfill, parse_ncu_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    # 这个脚本的输入是“一个 microbench JSON + 一份 NCU 导出”。
    # 为了防止 CSV / .ncu-rep 同时传入造成歧义，这里用互斥组强制二选一。
    parser = argparse.ArgumentParser(description="Backfill measured ncu metrics into microbench JSON")
    parser.add_argument("json", type=Path, help="microbench 输出的 JSON")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ncu-csv", type=Path, help="ncu --csv 导出的 CSV 文件")
    src.add_argument("--ncu-rep", type=Path, help="ncu 的 .ncu-rep 报告（自动转 CSV）")
    parser.add_argument("--ncu-bin", default="ncu", help="ncu 可执行文件名/路径")
    parser.add_argument("--output", type=Path, help="输出 JSON（默认原地覆盖）")
    return parser.parse_args()


def _csv_from_rep(rep: Path, ncu_bin: str) -> str:
    """用 ncu --import 把 .ncu-rep 转成 CSV 文本。"""

    # 这里不自己解析 .ncu-rep，而是让 ncu 官方工具先转换成 CSV，保证格式一致。
    cmd = [ncu_bin, "--import", str(rep), "--csv", "--page", "raw"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ncu --import 失败（退出码 {proc.returncode}）：\n{proc.stderr}")
    return proc.stdout


def main() -> None:
    # 1) 读入 NCU 数据
    # 2) 解析出我们关心的 measured 指标
    # 3) 回填到原始 microbench JSON
    # 4) 原地覆盖或写入新文件
    args = parse_args()
    if args.ncu_csv:
        # CSV 已经准备好时，直接读文本最省事，也方便在 notebook 里串联。
        csv_text = args.ncu_csv.read_text(encoding="utf-8", errors="replace")
    else:
        # 如果是 .ncu-rep，则先调用 ncu 做一次转换。
        csv_text = _csv_from_rep(args.ncu_rep, args.ncu_bin)

    metrics = parse_ncu_csv(csv_text)
    data = json.loads(args.json.read_text(encoding="utf-8"))
    # 回填 measured_* 字段 + bandwidth_fields_are_estimates + profiler_warning（与 microbench 共用）。
    bad = apply_backfill(data, metrics)
    if bad:
        # 采样到不该 profile 的 kernel 时，额外在 stdout 给出更详细的逐条提示。
        print("[警告] 疑似 profile 了非 attention kernel（量化/elementwise），实测字节可能偏大：")
        for n in sorted(set(bad))[:5]:
            print(f"    - {n}")
        print("  建议用 --kernel-name regex 只 profile attention kernel 后重新采集。")

    est = data.get("estimated_effective_quant_kv_bandwidth_gbps")
    meas = metrics.achieved_bandwidth_gbps
    out = args.output or args.json
    # 默认覆盖原 JSON，因为这个脚本的主要作用就是给已有结果补 measured 字段。
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 打印一组最关键的对比值，方便在 shell / notebook 里快速确认回填是否合理。
    print(f"回填 {out}")
    print(f"  kernels profiled     : {metrics.kernel_count}")
    print(f"  measured DRAM bytes  : {metrics.dram_bytes:,.0f}")
    print(f"  measured bandwidth   : {meas:,.1f} GB/s")
    if est:
        print(f"  estimated bandwidth  : {est:,.1f} GB/s  (measured/estimated = {meas/est:.2f}x)")
    if metrics.occupancy_pct is not None:
        print(f"  occupancy            : {metrics.occupancy_pct:.1f} %")
    if metrics.sm_utilization_pct is not None:
        print(f"  SM utilization       : {metrics.sm_utilization_pct:.1f} %")


if __name__ == "__main__":
    main()
