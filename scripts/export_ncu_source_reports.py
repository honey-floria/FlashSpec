from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # 这个脚本只做一件事：把 .ncu-rep 批量导出成可读的 txt/csv。
    # 这样 source attribution 报告就能直接吃导出的文本，而不必依赖 GUI。
    parser = argparse.ArgumentParser(description="Export Nsight Compute .ncu-rep reports for source attribution analysis")
    parser.add_argument("--input-dir", type=Path, default=Path("results/ncu_source_attribution"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ncu_source_attribution_export"))
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument(
        "--pages",
        default="details,raw,source",
        help="Comma-separated ncu report pages to export. Common values: details,raw,source,session",
    )
    return parser.parse_args()


def export_page(ncu_bin: str, rep: Path, page: str, output: Path) -> bool:
    # raw page 适合后续做 CSV 解析；details/source 则保留为 txt 供人工阅读。
    cmd = [ncu_bin, "--import", str(rep), "--page", page]
    if page == "raw":
        cmd.append("--csv")
        output = output.with_suffix(".csv")
    else:
        output = output.with_suffix(".txt")

    # ncu 的 stdout 直接写入导出文件；如果命令失败，再单独写 stderr，方便定位问题。
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        err = output.with_suffix(output.suffix + ".stderr.txt")
        err.write_text(proc.stderr, encoding="utf-8", errors="replace")
        print(f"warning: ncu export failed for {rep.name} page={page}; stderr written to {err}")
        return False
    return True


def main() -> None:
    # 遍历 input-dir 下的所有 .ncu-rep，然后按 pages 参数逐页导出。
    args = parse_args()
    reports = sorted(args.input_dir.glob("*.ncu-rep"))
    if not reports:
        raise FileNotFoundError(f"No .ncu-rep files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = [page.strip() for page in args.pages.split(",") if page.strip()]
    failures = 0
    for rep in reports:
        for page in pages:
            out = args.output_dir / f"{rep.stem}.{page}"
            print(f"export {rep.name} page={page} -> {out}")
            if not export_page(args.ncu_bin, rep, page, out):
                failures += 1
    if failures:
        # 不中断于单个失败，全部导出完再统一返回非零，方便一次性看到缺哪些页。
        raise SystemExit(f"completed with {failures} failed exports")


if __name__ == "__main__":
    main()
