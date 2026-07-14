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
    fused_dequant_attention_triton,
    paged_quant_attention,
    paged_quant_attention_triton,
    quantize_int8_per_block,
    reference_attention,
)
from flashspec.triton_kernels import HAS_TRITON


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

    def test_triton_fused_wrapper_falls_back_on_cpu(self) -> None:
        q = torch.randn(2, 3, 16)
        k = torch.randn(2, 3, 25, 16)
        v = torch.randn(2, 3, 25, 16)
        kq = quantize_int8_per_block(k, block_size=8)
        vq = quantize_int8_per_block(v, block_size=8)
        expected = fused_dequant_attention(q, kq, vq)
        actual, stats = fused_dequant_attention_triton(q, kq, vq, return_stats=True)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
        self.assertEqual(stats["materializes_dense_kv"], 1.0)

    @unittest.skipUnless(HAS_TRITON and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_fused_attention_matches_dequant_reference_on_cuda(self) -> None:
        device = torch.device("cuda")
        dtype = torch.float16
        q = torch.randn(2, 3, 64, device=device, dtype=dtype)
        k = torch.randn(2, 3, 129, 64, device=device, dtype=dtype)
        v = torch.randn(2, 3, 129, 64, device=device, dtype=dtype)
        lengths = torch.tensor([129, 113], device=device, dtype=torch.int64)
        kq = quantize_int8_per_block(k, block_size=16)
        vq = quantize_int8_per_block(v, block_size=16)
        expected = reference_attention(q, dequantize_int8_per_block(kq), dequantize_int8_per_block(vq), lengths=lengths)
        actual, stats = fused_dequant_attention_triton(q, kq, vq, lengths=lengths, return_stats=True)
        torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
        self.assertEqual(stats["materializes_dense_kv"], 0.0)

    @unittest.skipUnless(HAS_TRITON and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_fused_split_k_matches_reference_on_cuda(self) -> None:
        # seq_len=1500 > TOKENS_PER_SPLIT(512) 强制 num_splits>1，确保 Split-K
        # 的 split+combine 路径被测到；variable lengths 覆盖尾段 mask。
        device = torch.device("cuda")
        dtype = torch.float16
        q = torch.randn(2, 4, 128, device=device, dtype=dtype)
        k = torch.randn(2, 4, 1500, 128, device=device, dtype=dtype)
        v = torch.randn(2, 4, 1500, 128, device=device, dtype=dtype)
        lengths = torch.tensor([1500, 977], device=device, dtype=torch.int64)
        kq = quantize_int8_per_block(k, block_size=16)
        vq = quantize_int8_per_block(v, block_size=16)
        expected = reference_attention(
            q, dequantize_int8_per_block(kq), dequantize_int8_per_block(vq), lengths=lengths
        )
        actual, stats = fused_dequant_attention_triton(q, kq, vq, lengths=lengths, return_stats=True)
        torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
        self.assertEqual(stats["materializes_dense_kv"], 0.0)
        self.assertGreater(stats["num_splits"], 1.0)

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

    def test_triton_paged_wrapper_falls_back_on_cpu(self) -> None:
        q = torch.randn(2, 2, 8)
        k = torch.randn(2, 2, 17, 8)
        v = torch.randn(2, 2, 17, 8)
        cache = PagedKVCache.from_dense(k, v, block_size=4)
        expected = paged_quant_attention(q, cache)
        actual, stats = paged_quant_attention_triton(q, cache, return_stats=True)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
        self.assertEqual(stats["materializes_dense_kv"], 1.0)

    @unittest.skipUnless(HAS_TRITON and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_paged_attention_matches_reconstructed_cache_on_cuda(self) -> None:
        device = torch.device("cuda")
        dtype = torch.float16
        q = torch.randn(2, 3, 64, device=device, dtype=dtype)
        k = torch.randn(2, 3, 129, 64, device=device, dtype=dtype)
        v = torch.randn(2, 3, 129, 64, device=device, dtype=dtype)
        cache = PagedKVCache.from_dense(k, v, block_size=16)
        variable_lengths = torch.tensor([129, 113], device=device, dtype=torch.int64)
        cache = PagedKVCache(
            k_quant=cache.k_quant,
            v_quant=cache.v_quant,
            block_table=cache.block_table,
            lengths=variable_lengths,
            block_size=cache.block_size,
            max_seq_len=cache.max_seq_len,
        )
        kd, vd = cache.to_dense()
        expected = reference_attention(q, kd, vd, lengths=cache.lengths)
        actual, stats = paged_quant_attention_triton(q, cache, return_stats=True)
        torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
        self.assertEqual(stats["materializes_dense_kv"], 0.0)

    def test_paged_cache_append_extends_cache_without_dense_rebuild(self) -> None:
        k = torch.randn(2, 2, 5, 8)
        v = torch.randn(2, 2, 5, 8)
        k_new = torch.randn(2, 2, 3, 8)
        v_new = torch.randn(2, 2, 3, 8)
        cache = PagedKVCache.from_dense(k, v, block_size=4)
        appended = cache.append(k_new, v_new)

        kd, vd = appended.to_dense()
        expected_k = torch.cat([k, k_new], dim=2)
        expected_v = torch.cat([v, v_new], dim=2)
        self.assertEqual(tuple(appended.lengths.tolist()), (8, 8))
        self.assertEqual(tuple(kd.shape), tuple(expected_k.shape))
        self.assertEqual(tuple(vd.shape), tuple(expected_v.shape))
        self.assertLess(float((expected_k - kd).abs().max().item()), 0.06)
        self.assertLess(float((expected_v - vd).abs().max().item()), 0.06)

    def test_paged_attention_after_append_matches_reconstructed_cache(self) -> None:
        q = torch.randn(2, 2, 8)
        k = torch.randn(2, 2, 7, 8)
        v = torch.randn(2, 2, 7, 8)
        k_new = torch.randn(2, 2, 2, 8)
        v_new = torch.randn(2, 2, 2, 8)
        cache = PagedKVCache.from_dense(k, v, block_size=4).append(k_new, v_new)
        kd, vd = cache.to_dense()

        expected = reference_attention(q, kd, vd, lengths=cache.lengths)
        actual = paged_quant_attention(q, cache)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)

    @unittest.skipUnless(HAS_TRITON and torch.cuda.is_available(), "requires Triton and CUDA")
    def test_triton_paged_attention_after_append_matches_reconstructed_cache_on_cuda(self) -> None:
        device = torch.device("cuda")
        dtype = torch.float16
        q = torch.randn(2, 2, 64, device=device, dtype=dtype)
        k = torch.randn(2, 2, 7, 64, device=device, dtype=dtype)
        v = torch.randn(2, 2, 7, 64, device=device, dtype=dtype)
        k_new = torch.randn(2, 2, 3, 64, device=device, dtype=dtype)
        v_new = torch.randn(2, 2, 3, 64, device=device, dtype=dtype)
        cache = PagedKVCache.from_dense(k, v, block_size=4).append(k_new, v_new)
        kd, vd = cache.to_dense()
        expected = reference_attention(q, kd, vd, lengths=cache.lengths)
        actual, stats = paged_quant_attention_triton(q, cache, return_stats=True)
        torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
        self.assertEqual(stats["materializes_dense_kv"], 0.0)


if __name__ == "__main__":
    unittest.main()
