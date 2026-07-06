from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep FlashSpec batch and sequence shapes")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--seq-lens", default="128,512,1024")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--output", type=Path, default=Path("results/sweep.csv"))
    return parser.parse_args()


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for backend in ("dense", "paged"):
        for batch in _ints(args.batches):
            for seq_len in _ints(args.seq_lens):
                command = [
                    sys.executable,
                    str(ROOT / "benchmarks" / "microbench.py"),
                    "--backend",
                    backend,
                    "--batch",
                    str(batch),
                    "--heads",
                    str(args.heads),
                    "--seq-len",
                    str(seq_len),
                    "--head-dim",
                    str(args.head_dim),
                    "--iters",
                    str(args.iters),
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--json",
                ]
                proc = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                import json

                rows.append({key: str(value) for key, value in json.loads(proc.stdout).items()})

    fieldnames = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
