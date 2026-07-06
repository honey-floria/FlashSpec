from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit a simple FlashSpec roofline SVG")
    parser.add_argument("--peak-tflops", type=float, default=65.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=320.0)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--achieved-tflops", type=float, default=0.32)
    parser.add_argument("--output", type=Path, default=Path("results/roofline.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ridge = args.peak_tflops * 1000.0 / args.bandwidth_gbps
    x = 80 + min(500, args.intensity / max(ridge, 1.0e-9) * 420)
    y = 300 - min(220, args.achieved_tflops / max(args.peak_tflops, 1.0e-9) * 220)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" fill="#fafafa"/>
  <text x="48" y="40" font-family="Arial" font-size="22" font-weight="700">FlashSpec Roofline Sketch</text>
  <line x1="80" y1="320" x2="640" y2="320" stroke="#333"/>
  <line x1="80" y1="320" x2="80" y2="70" stroke="#333"/>
  <text x="300" y="365" font-family="Arial" font-size="14">Arithmetic intensity, FLOP/byte</text>
  <text x="18" y="210" font-family="Arial" font-size="14" transform="rotate(-90 18 210)">Performance, TFLOP/s</text>
  <polyline points="80,320 520,90 640,90" fill="none" stroke="#666" stroke-width="2"/>
  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#d04a02"/>
  <text x="{x + 12:.1f}" y="{y - 8:.1f}" font-family="Arial" font-size="13">decode attention</text>
  <text x="92" y="96" font-family="Arial" font-size="13">Peak compute: {args.peak_tflops:.1f} TFLOP/s</text>
  <text x="92" y="116" font-family="Arial" font-size="13">HBM bandwidth: {args.bandwidth_gbps:.1f} GB/s</text>
  <text x="92" y="136" font-family="Arial" font-size="13">Ridge point: {ridge:.1f} FLOP/byte</text>
  <text x="92" y="156" font-family="Arial" font-size="13">Decode intensity: {args.intensity:.2f} FLOP/byte</text>
</svg>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

