from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.flashspec import PagedKVCache, paged_quant_attention


def main() -> None:
    q = torch.randn(2, 4, 32)
    k = torch.randn(2, 4, 128, 32)
    v = torch.randn(2, 4, 128, 32)
    cache = PagedKVCache.from_dense(k, v, block_size=16)
    out, stats = paged_quant_attention(q, cache, return_stats=True)
    print(out.shape)
    print(stats)


if __name__ == "__main__":
    main()

