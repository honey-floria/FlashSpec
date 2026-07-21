from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from flashspec import (
    PagedKVAllocator,
    PagedKVCache,
    dequantize_int8_per_block,
    fused_dequant_attention,
    fused_dequant_attention_triton,
    paged_quant_attention,
    paged_quant_attention_triton,
    quantize_int8_per_block,
    reference_attention,
)
import flashspec.triton_fused as triton_fused
import flashspec.triton_paged as triton_paged
from flashspec.triton_utils import HAS_TRITON


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

    def test_triton_default_knobs_match_profiled_best(self) -> None:
        old = {key: os.environ.get(key) for key in ("FLASHSPEC_NUM_SPLITS", "FLASHSPEC_BLOCK_N", "FLASHSPEC_NUM_WARPS")}
        try:
            for key in old:
                os.environ.pop(key, None)

            self.assertEqual(triton_fused._resolve_block_n(), 128)
            self.assertEqual(triton_fused._resolve_num_warps(), 4)
            self.assertEqual(triton_fused._compute_num_splits(seq_len=2048, block_n=128), 4)
            self.assertEqual(triton_fused._compute_num_splits(seq_len=4096, block_n=128), 4)
            self.assertEqual(triton_paged._resolve_block_n(), 128)
            self.assertEqual(triton_paged._resolve_num_warps(), 4)

            os.environ["FLASHSPEC_BLOCK_N"] = "64"
            os.environ["FLASHSPEC_NUM_WARPS"] = "8"
            os.environ["FLASHSPEC_NUM_SPLITS"] = "1"
            self.assertEqual(triton_fused._resolve_block_n(), 64)
            self.assertEqual(triton_fused._resolve_num_warps(), 8)
            self.assertEqual(triton_fused._compute_num_splits(seq_len=2048, block_n=64), 1)
            self.assertEqual(triton_paged._resolve_block_n(), 64)
            self.assertEqual(triton_paged._resolve_num_warps(), 8)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

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

    def test_paged_cache_profile_layouts_preserve_logical_kv(self) -> None:
        k = torch.randn(3, 2, 17, 8)
        v = torch.randn(3, 2, 17, 8)
        lengths = torch.tensor([17, 11, 5], dtype=torch.int64)
        for layout in ("shuffled", "interleaved"):
            cache = PagedKVCache.from_dense(
                k,
                v,
                block_size=4,
                lengths=lengths,
                block_table_pattern=layout,
                layout_seed=13,
            )
            kd, vd = cache.to_dense()
            self.assertEqual(tuple(cache.lengths.tolist()), tuple(lengths.tolist()))
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

    def test_paged_allocator_append_release_and_reuse(self) -> None:
        allocator = PagedKVAllocator(capacity_blocks=6, heads=2, head_dim=8, block_size=4)
        k0 = torch.randn(1, 2, 5, 8)
        v0 = torch.randn(1, 2, 5, 8)
        k1 = torch.randn(1, 2, 3, 8)
        v1 = torch.randn(1, 2, 3, 8)
        allocator.add_request(10, k0, v0)
        allocator.add_request(20, k1, v1)

        initial_cache = allocator.to_cache([10, 20])
        released_physical_block = int(initial_cache.block_table[1, 0].item())
        self.assertEqual(allocator.stats()["allocated_blocks"], 3.0)

        k_new = torch.randn(1, 2, 4, 8)
        v_new = torch.randn(1, 2, 4, 8)
        allocator.append_request(10, k_new, v_new)
        appended = allocator.to_cache([10])
        kd, vd = appended.to_dense()
        expected_k = torch.cat([k0, k_new], dim=2)
        expected_v = torch.cat([v0, v_new], dim=2)
        self.assertEqual(tuple(appended.lengths.tolist()), (9,))
        self.assertEqual(allocator.stats()["allocated_blocks"], 4.0)
        self.assertLess(float((expected_k - kd).abs().max().item()), 0.08)
        self.assertLess(float((expected_v - vd).abs().max().item()), 0.08)

        allocator.release_request(20)
        k2 = torch.randn(1, 2, 1, 8)
        v2 = torch.randn(1, 2, 1, 8)
        allocator.add_request(30, k2, v2)
        reused = allocator.to_cache([30])
        self.assertEqual(int(reused.block_table[0, 0].item()), released_physical_block)
        self.assertIn("fragmentation", allocator.stats())

    def test_microbench_json_schema_includes_experiment_knobs(self) -> None:
        env = os.environ.copy()
        env.pop("FLASHSPEC_NUM_SPLITS", None)
        env.pop("FLASHSPEC_BLOCK_N", None)
        env.pop("FLASHSPEC_NUM_WARPS", None)
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "microbench.py"),
                "--backend", "dense",
                "--batch", "1",
                "--heads", "1",
                "--seq-len", "4",
                "--head-dim", "8",
                "--iters", "1",
                "--warmup", "0",
                "--repeats", "1",
                "--device", "cpu",
                "--dtype", "float32",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        data = json.loads(proc.stdout)
        for key in (
            "backend",
            "seq_len",
            "head_dim",
            "block_size",
            "num_splits",
            "block_n",
            "env_flashspec_num_splits",
            "env_flashspec_block_n",
            "env_flashspec_num_warps",
            "num_warps",
            "length_pattern",
            "passes_lengths_to_attention",
            "effective_lengths",
            "effective_min_seq_len",
            "effective_max_seq_len",
            "paged_layout",
            "paged_layout_seed",
            "bandwidth_fields_are_estimates",
            "measured_registers_per_thread",
            "measured_theoretical_occupancy_pct",
            "nsight_compute_source_command",
            "nsight_compute_commands",
            "materializes_dense_kv",
        ):
            self.assertIn(key, data)
        self.assertIsNone(data["block_n"])
        self.assertIsNone(data["num_warps"])
        self.assertIsNone(data["env_flashspec_num_splits"])
        self.assertIsNone(data["env_flashspec_block_n"])
        self.assertIsNone(data["env_flashspec_num_warps"])
        self.assertEqual(data["length_pattern"], "uniform")
        self.assertFalse(data["passes_lengths_to_attention"])
        self.assertEqual(data["effective_lengths"], [4])
        self.assertEqual(data["paged_layout"], "contiguous")

    def test_serving_json_schema_includes_latency_and_allocator_fields(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "e2e_serving.py"),
                "--requests",
                "2",
                "--prompt-lens",
                "3,5",
                "--prompt-length-distribution",
                "bimodal",
                "--decode-steps",
                "3",
                "--request-life-steps",
                "2",
                "--heads",
                "1",
                "--head-dim",
                "8",
                "--block-size",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(proc.stdout)
        for key in (
            "ttft_ms",
            "tpot_ms",
            "tokens_per_second",
            "prefill_ms",
            "decode_ms",
            "block_utilization",
            "fragmentation",
            "allocated_blocks",
            "free_blocks",
            "live_requests",
            "used_tokens",
            "capacity_tokens",
            "padding_waste",
            "arrivals",
            "finishes",
            "prompt_lens",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["prompt_lens"], [3, 5])
        self.assertGreaterEqual(data["finishes"], 2)

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


class NcuBackfillTest(unittest.TestCase):
    """覆盖 microbench 和 backfill_ncu 共用的 ncu 回填 helper（纯 CPU，无需 CUDA）。"""

    _CSV = (
        "==PROF== banner line to skip\n"
        '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"0","fused_dequant_attention_kernel","dram__bytes_read.sum","1000","byte"\n'
        '"0","fused_dequant_attention_kernel","gpu__time_duration.sum","10","ns"\n'
    )

    def test_apply_backfill_writes_measured_fields_and_clears_estimate_flag(self) -> None:
        from scripts.ncu_parse import apply_backfill, parse_ncu_csv

        result = {"bandwidth_fields_are_estimates": True}
        bad = apply_backfill(result, parse_ncu_csv(self._CSV))
        self.assertEqual(bad, [])
        self.assertFalse(result["bandwidth_fields_are_estimates"])
        self.assertEqual(result["measured_dram_bytes"], 1000.0)
        self.assertEqual(result["measured_ncu_kernel_count"], 1)
        self.assertNotIn("profiler_warning", result)

    def test_apply_backfill_flags_suspicious_kernels(self) -> None:
        from scripts.ncu_parse import apply_backfill, parse_ncu_csv

        csv_text = self._CSV.replace("fused_dequant_attention_kernel", "vectorized_elementwise_kernel")
        result: dict = {}
        bad = apply_backfill(result, parse_ncu_csv(csv_text))
        self.assertTrue(bad)
        self.assertIn("profiler_warning", result)


if __name__ == "__main__":
    unittest.main()
