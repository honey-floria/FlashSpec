"""FlashSpec 逐步 walkthrough —— 带 print 的可运行脚本。

运行：
    python examples/walkthrough.py

目的：不是跑 benchmark，而是把 quant -> paged -> attention 这条链路里
每一步的 tensor 形状和数值都打印出来，让你「看得见」代码在做什么。
所有 shape 都故意取得很小，方便你把数字和代码对上。

建议配合以下源码一起看（难度从低到高）：
    src/flashspec/quant.py       量化 / 反量化
    src/flashspec/paged.py       paged KV cache
    src/flashspec/attention.py   attention
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 让脚本无论从哪运行都能 import 到 flashspec。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flashspec import (
    PagedKVCache,
    dequantize_int8_per_block,
    paged_quant_attention,
    quantize_int8_per_block,
    reference_attention,
)


def section(title: str) -> None:
    """打印一个醒目的分节标题，方便在输出里定位。"""
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def show(name: str, t: torch.Tensor) -> None:
    """打印 tensor 的形状、dtype 和数值。"""
    print(f"\n{name}: shape={tuple(t.shape)}, dtype={t.dtype}")
    print(t)


def part1_quantization() -> None:
    """第 1 部分：INT8 量化 / 反量化（对应 quant.py）。

    把一段浮点数在每个 block 内找 min/max，线性映射到 [0,255] 存成 int8，
    反量化再映射回浮点。
    """
    section("PART 1  量化 round trip  (quant.py)")

    torch.manual_seed(0)
    # 形状 [batch, heads, seq_len, head_dim]，取超小值方便看。
    x = torch.randn(1, 1, 4, 2)
    show("原始浮点 x", x)

    # block_size=2: seq 维每 2 个 token 一个量化 block；seq_len=4 -> 2 个 block。
    # 每个 block 内 [block_size, head_dim]=[2,2] 共享一组 scale/zero_point。
    q = quantize_int8_per_block(x, block_size=2)
    show("量化 values (int8, 实际存 uint8-128)", q.values)
    show("每个 block 的 scale", q.scale)
    show("每个 block 的 zero_point", q.zero_point)
    print(f"\nvalues.shape={tuple(q.values.shape)} <- [batch, heads, n_blocks, block_size, head_dim]")

    # 反量化：x' = (values+128 - zero_point) * scale
    x_rec = dequantize_int8_per_block(q)
    show("反量化还原 x'", x_rec)
    print(f"\n最大还原误差 = {(x - x_rec).abs().max().item():.4f}  (INT8 有损但很小)")


def part2_paged_cache() -> None:
    """第 2 部分：paged cache 的 from_dense / to_dense（对应 paged.py）。

    KV 不连续存，而是切成固定大小 block；block_table 记录逻辑->物理映射。
    """
    section("PART 2  paged cache 切块与还原  (paged.py)")

    torch.manual_seed(1)
    # seq_len=3, block_size=2 -> 2 个 block，最后一个只用 1 个位置(补零)。
    k = torch.randn(1, 1, 3, 2)
    v = torch.randn(1, 1, 3, 2)
    show("dense k [batch, heads, seq=3, dim=2]", k)

    cache = PagedKVCache.from_dense(k, v, block_size=2)
    print(f"\nmax_seq_len = {cache.max_seq_len}  (真实有效长度)")
    show("block_table [batch, logical_blocks]", cache.block_table)
    show("lengths [batch]", cache.lengths)
    show("物理 K block int8 values", cache.k_quant.values)
    print(f"物理 K values.shape = {tuple(cache.k_quant.values.shape)}")
    print("  ^ [physical_blocks, heads, block_size, head_dim]；seq=3 补零到 4 = 2 block x 2")

    kd, vd = cache.to_dense()
    show("to_dense() 还原的 k", kd)
    print(f"\n还原 shape={tuple(kd.shape)}，已裁回 seq=3（padding 位置被丢弃）")
    print(f"和原始 k 最大误差 = {(k - kd).abs().max().item():.4f}")


def part3_append() -> None:
    """第 3 部分：往 cache append 新 token（decode 的核心动作）。"""
    section("PART 3  append 一个新 token  (paged.py)")

    torch.manual_seed(2)
    k = torch.randn(1, 1, 3, 2)
    v = torch.randn(1, 1, 3, 2)
    cache = PagedKVCache.from_dense(k, v, block_size=2)
    print(f"append 前: lengths={cache.lengths.tolist()}, table 列数={cache.block_table.shape[1]}")

    # 新增 1 个 token；原 seq=3，第 4 个 token 落在 logical block 1 的 offset 1。
    k_new = torch.randn(1, 1, 1, 2)
    v_new = torch.randn(1, 1, 1, 2)
    cache2 = cache.append(k_new, v_new)
    print(f"append 后: lengths={cache2.lengths.tolist()}, table 列数={cache2.block_table.shape[1]}")

    kd, _ = cache2.to_dense()
    expected = torch.cat([k, k_new], dim=2)
    show("append 后还原的 k（应含 4 个 token）", kd)
    print(f"\n和 [老k; 新k] 拼接的最大误差 = {(expected - kd).abs().max().item():.4f}")


def part4_attention() -> None:
    """第 4 部分：两条路径给出同样结果（对应 attention.py）。"""
    section("PART 4  attention 路径对比  (attention.py)")

    torch.manual_seed(3)
    # q 只有一个 decode token: [batch, heads, head_dim]，没有 seq 维。
    q = torch.randn(1, 2, 2)
    k = torch.randn(1, 2, 3, 2)
    v = torch.randn(1, 2, 3, 2)
    show("q [batch, heads, head_dim]", q)
    print("注意 q 没有 seq 维 —— decode 一次只算一个新 token。")

    # A. 标准 dense attention（标准答案）
    out_ref = reference_attention(q, k, v)
    show("A. reference_attention 输出", out_ref)

    # B. 走 paged quant cache
    cache = PagedKVCache.from_dense(k, v, block_size=2)
    out_paged = paged_quant_attention(q, cache)
    show("B. paged_quant_attention 输出", out_paged)

    print(f"\nA vs B 最大差异 = {(out_ref - out_paged).abs().max().item():.4f}")
    print("差异只来自 INT8 量化误差；attention 逻辑本身完全一致。")
    print("输出 shape = q 的 shape = [batch, heads, head_dim]，就是这个 token 的结果。")


def main() -> None:
    part1_quantization()
    part2_paged_cache()
    part3_append()
    part4_attention()
    section("完成")
    print("建议：改一改上面的 shape / block_size / seed 再跑，看数字怎么变。")


if __name__ == "__main__":
    main()
