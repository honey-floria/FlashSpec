from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flashspec.serving import ServingConfig, run_decode_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashSpec serving-loop simulation")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_decode_simulation(
        ServingConfig(
            requests=args.requests,
            prompt_len=args.prompt_len,
            decode_steps=args.decode_steps,
            heads=args.heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            device=args.device,
            dtype=args.dtype,
        )
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
