from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flashspec import (
    PagedKVCache,
    dequantize_int8_per_block,
    fused_dequant_attention,
    paged_quant_attention,
    quantize_int8_per_block,
    reference_attention,
)


class FlashSpecTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_quant_round_trip_has_bounded_error(self) -> None:
        x = torch.randn(2, 3, 19, 16)
        q = quantize_int8_per_block(x, block_size=8)
        y = dequantize_int8_per_block(q)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertLess(float((x - y).abs().max().item()), 0.03)

    def test_fused_dequant_attention_matches_dequant_reference(self) -> None:
        q = torch.randn(2, 3, 16)
        k = torch.randn(2, 3, 25, 16)
        v = torch.randn(2, 3, 25, 16)
        kq = quantize_int8_per_block(k, block_size=8)
        vq = quantize_int8_per_block(v, block_size=8)
        expected = reference_attention(q, dequantize_int8_per_block(kq), dequantize_int8_per_block(vq))
        actual = fused_dequant_attention(q, kq, vq)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)

    def test_paged_cache_reconstructs_dense_quantized_kv(self) -> None:
        k = torch.randn(2, 4, 23, 8)
        v = torch.randn(2, 4, 23, 8)
        cache = PagedKVCache.from_dense(k, v, block_size=6)
        kd, vd = cache.to_dense()
        self.assertEqual(tuple(kd.shape), tuple(k.shape))
        self.assertEqual(tuple(vd.shape), tuple(v.shape))
        self.assertLess(float((k - kd).abs().max().item()), 0.03)
        self.assertLess(float((v - vd).abs().max().item()), 0.03)

    def test_paged_attention_matches_dense_quant_attention(self) -> None:
        q = torch.randn(2, 2, 8)
        k = torch.randn(2, 2, 17, 8)
        v = torch.randn(2, 2, 17, 8)
        cache = PagedKVCache.from_dense(k, v, block_size=4)
        kd, vd = cache.to_dense()
        expected = reference_attention(q, kd, vd, lengths=cache.lengths)
        actual = paged_quant_attention(q, cache)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()

