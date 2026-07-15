from __future__ import annotations

from typing import Any

import math
import os

import torch

from .paged import PagedKVCache
from .triton_utils import HAS_TRITON, next_power_of_2, tl, triton


if HAS_TRITON:

    @triton.jit
    def _paged_quant_attention_kernel(
        q_ptr,
        k_values_ptr,
        k_scale_ptr,
        k_zero_point_ptr,
        v_values_ptr,
        v_scale_ptr,
        v_zero_point_ptr,
        block_table_ptr,
        lengths_ptr,
        out_ptr,
        heads: tl.constexpr,
        max_seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        table_cols: tl.constexpr,
        physical_quant_blocks: tl.constexpr,
        page_block_size: tl.constexpr,
        quant_block_size: tl.constexpr,
        sm_scale: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ) -> None:
        """Triton Kernel 2：直接按 block_table 读取 paged INT8 KV 并执行 attention。

        每个 Triton program 负责一个 [batch, head] 的 decode attention。
        与 Kernel 1 的区别是：K/V 不再按 dense sequence 顺序连续存储，而是
        被拆成 physical block。kernel 必须先把 token 位置映射成 logical block，
        再通过 block_table 查到 physical block，最后从该 physical block 中读取
        INT8 K/V。

        关键点：
        - 不调用 cache.to_dense()。
        - 不 gather 出完整 dense K/V。
        - 在 kernel 内完成 block_table 间接寻址、INT8 反量化和 attention。
        """

        # pid 对应展平后的 [batch, head] 编号。
        pid = tl.program_id(0)

        # 从 pid 还原 batch 下标和 head 下标。
        batch_idx = pid // heads
        head_idx = pid - batch_idx * heads

        # 当前 head 内的 feature/head_dim 下标。
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        # 读取当前 [batch, head] 的 query 向量，布局为 [batch, heads, head_dim]。
        q_base = (batch_idx * heads + head_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        # 每条 request 可以有不同有效长度。effective_len 决定 attention 可见 token 范围。
        loaded_len = tl.load(lengths_ptr + batch_idx)
        effective_len = tl.minimum(loaded_len, max_seq_len)

        # online softmax 状态，避免保存完整 scores。
        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 按逻辑 sequence 位置扫描；每个 token 再通过 block_table 映射到物理 block。
        for start in range(0, max_seq_len, block_n):
            offs_s = start + tl.arange(0, block_n)
            s_mask = offs_s < effective_len

            # logical_block 是 token 属于该 request 的第几个逻辑页。
            # page_offset 是 token 在该页内的位置。
            logical_block = offs_s // page_block_size
            page_offset = offs_s - logical_block * page_block_size

            # block_table 布局是 [batch, table_cols]。
            # 未分配的 logical block 通常是 -1；有效长度 mask 会避免访问无效 token。
            table_mask = s_mask & (logical_block < table_cols)
            physical_block = tl.load(
                block_table_ptr + batch_idx * table_cols + logical_block,
                mask=table_mask,
                other=0,
            )
            physical_valid = table_mask & (physical_block >= 0)

            # physical block 内部的 sequence 位置还要映射到量化 block。
            # 当前 PagedKVCache 默认 page_block_size == quant_block_size，因此
            # physical_quant_block 通常为 0；这里保留通用写法，方便后续扩展。
            physical_quant_block = page_offset // quant_block_size
            quant_offset = page_offset - physical_quant_block * quant_block_size
            qparam_mask = physical_valid & (physical_quant_block < physical_quant_blocks)

            # QuantizedTensor.values 物理布局：
            # [physical_blocks, heads, physical_quant_blocks, quant_block_size, head_dim]。
            kv_offsets = (
                ((((physical_block[:, None] * heads + head_idx) * physical_quant_blocks + physical_quant_block[:, None])
                  * quant_block_size + quant_offset[:, None])
                 * head_dim)
                + offs_d[None, :]
            )
            kv_mask = qparam_mask[:, None] & d_mask[None, :]

            # scale/zero_point 布局是 [physical_blocks, heads, physical_quant_blocks, 1, 1]。
            qparam_offsets = (physical_block * heads + head_idx) * physical_quant_blocks + physical_quant_block
            k_scale = tl.load(k_scale_ptr + qparam_offsets, mask=qparam_mask, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_point_ptr + qparam_offsets, mask=qparam_mask, other=0).to(tl.float32)

            # INT8 values 以 signed int8 存储，反量化时先 +128 回到 uint8 域。
            k_i8 = tl.load(k_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
            k_deq = (k_i8 + 128.0 - k_zero[:, None]) * k_scale[:, None]

            # QK 点积得到当前 block_n 个 token 的 attention scores。
            scores = tl.sum(k_deq * q[None, :], axis=1) * sm_scale
            scores = tl.where(qparam_mask, scores, -3.4028234663852886e38)

            # online softmax 更新：只保留当前最大值、分母和 PV 累积。
            block_m = tl.max(scores, axis=0)
            new_m = tl.maximum(m, block_m)
            old_scale = tl.exp(m - new_m)
            probs = tl.exp(scores - new_m)
            probs = tl.where(qparam_mask, probs, 0.0)
            new_l = l * old_scale + tl.sum(probs, axis=0)

            # V 走同样的 paged 间接寻址和 INT8 反量化。
            v_scale = tl.load(v_scale_ptr + qparam_offsets, mask=qparam_mask, other=1.0).to(tl.float32)
            v_zero = tl.load(v_zero_point_ptr + qparam_offsets, mask=qparam_mask, other=0).to(tl.float32)
            v_i8 = tl.load(v_values_ptr + kv_offsets, mask=kv_mask, other=-128).to(tl.float32)
            v_deq = (v_i8 + 128.0 - v_zero[:, None]) * v_scale[:, None]

            # 累积 PV。acc 始终和当前 new_m 对齐。
            acc = acc * old_scale + tl.sum(probs[:, None] * v_deq, axis=0)
            m = new_m
            l = new_l

        # 空长度 request 输出 0，避免除零。
        denom = tl.where(l > 0.0, l, 1.0)
        out = acc / denom
        out = tl.where(l > 0.0, out, 0.0)

        # 输出布局是 [batch, heads, head_dim]。
        tl.store(out_ptr + q_base + offs_d, out, mask=d_mask)


def _validate_triton_paged_inputs(q: torch.Tensor, cache: PagedKVCache) -> None:
    """校验 Triton paged attention 的输入约束。

    Kernel 2 当前实现聚焦 paged decode attention 主路径：
    - q 必须是 [batch, heads, head_dim]；
    - cache 内的 K/V 必须是 PagedKVCache 的物理 block 布局；
    - q、量化 K/V、block_table、lengths 必须位于同一个 CUDA 设备；
    - head_dim 当前限制为不超过 256，覆盖常见 64/128 配置。
    """

    if q.ndim != 3:
        raise ValueError("q 必须是 [batch, heads, head_dim] 形状")
    if q.shape[0] != cache.batch_size or q.shape[1] != cache.num_heads or q.shape[2] != cache.head_dim:
        raise ValueError("q 的 batch、heads 和 head_dim 必须与 PagedKVCache 对齐")
    if cache.k_quant.original_shape != cache.v_quant.original_shape:
        raise ValueError("cache.k_quant 和 cache.v_quant 的 original_shape 必须一致")
    if len(cache.k_quant.original_shape) != 4:
        raise ValueError("PagedKVCache 的 k_quant/v_quant 必须来自 [physical, heads, block, head_dim] 张量")
    if cache.k_quant.original_shape[2] != cache.block_size:
        raise ValueError("PagedKVCache 物理 block 的 sequence 维必须等于 cache.block_size")
    if cache.k_quant.block_size != cache.v_quant.block_size:
        raise ValueError("K/V 量化 block_size 必须一致")
    if cache.block_size % cache.k_quant.block_size != 0:
        raise ValueError("cache.block_size 必须能被量化 block_size 整除")
    if q.device.type != "cuda":
        raise RuntimeError("Triton paged attention 需要 CUDA q tensor")
    if cache.k_quant.device != q.device or cache.v_quant.device != q.device:
        raise RuntimeError("q 和 cache 的量化 K/V 必须位于同一个 CUDA 设备")
    if cache.block_table.device != q.device or cache.lengths.device != q.device:
        raise RuntimeError("block_table、lengths 必须与 q 位于同一个 CUDA 设备")
    if q.shape[-1] > 256:
        raise ValueError("Triton paged attention 当前只支持 head_dim <= 256")


def _resolve_block_n() -> int:
    """Kernel 2 每轮扫描的 logical token 数。默认 64，可用环境变量做 profiling sweep。"""

    override = os.environ.get("FLASHSPEC_BLOCK_N")
    if override is not None:
        try:
            v = int(override)
        except ValueError:
            v = 0
        if v in (16, 32, 64, 128):
            return v
    return 64


def _resolve_num_warps() -> int:
    """Triton program 的 warp 数。默认 4，可用环境变量做 profiling sweep。"""

    override = os.environ.get("FLASHSPEC_NUM_WARPS")
    if override is not None:
        try:
            v = int(override)
        except ValueError:
            v = 0
        if v in (1, 2, 4, 8):
            return v
    return 4


def _run_paged_quant_attention_triton(
    q: torch.Tensor,
    cache: PagedKVCache,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """运行真正的 Triton Kernel 2，不调用 cache.to_dense()。

    Python 侧只负责整理指针、分配输出和发起 kernel launch。kernel 内部会：
    1. 读取 block_table，完成 logical block -> physical block 的间接寻址。
    2. 从 physical INT8 K/V block 直接 load。
    3. 在寄存器中反量化。
    4. 完成 QK、online softmax 和 PV。

    因此主路径不会 gather 或 materialize 完整 dense KV。
    """

    _validate_triton_paged_inputs(q, cache)

    # contiguous 只整理现有元数据和 INT8 block store 的内存布局；
    # 不会把 paged cache 还原成 dense K/V。
    q_contig = q.contiguous()
    k_values = cache.k_quant.values.contiguous()
    k_scale = cache.k_quant.scale.contiguous()
    k_zero = cache.k_quant.zero_point.contiguous()
    v_values = cache.v_quant.values.contiguous()
    v_scale = cache.v_quant.scale.contiguous()
    v_zero = cache.v_quant.zero_point.contiguous()
    block_table = cache.block_table.contiguous()
    lengths = cache.lengths.contiguous()

    batch, heads, head_dim = q_contig.shape
    max_seq_len = int(cache.max_seq_len)
    table_cols = int(block_table.shape[1])
    physical_quant_blocks = int(k_values.shape[-3])
    page_block_size = int(cache.block_size)
    quant_block_size = int(cache.k_quant.block_size)
    block_d = next_power_of_2(int(head_dim))

    # 和 Kernel 1 保持一致，默认每次扫描 64 个逻辑 token；profiling 可用
    # FLASHSPEC_BLOCK_N 覆盖，用于观察 block_table 间接寻址下的 tile 取舍。
    block_n = _resolve_block_n()
    num_warps = _resolve_num_warps()
    out = torch.empty_like(q_contig)

    grid = (batch * heads,)
    _paged_quant_attention_kernel[grid](
        q_contig,
        k_values,
        k_scale,
        k_zero,
        v_values,
        v_scale,
        v_zero,
        block_table,
        lengths,
        out,
        heads,
        max_seq_len,
        head_dim,
        table_cols,
        physical_quant_blocks,
        page_block_size,
        quant_block_size,
        1.0 / math.sqrt(float(head_dim)),
        block_n,
        block_d,
        num_warps=num_warps,
    )

    if not return_stats:
        return out

    dense_bytes = 2 * batch * heads * max_seq_len * head_dim * q_contig.element_size()
    quant_bytes = cache.estimated_bytes()
    stats = {
        "dense_kv_bytes": float(dense_bytes),
        "quant_kv_bytes": float(quant_bytes),
        "compression_ratio": float(dense_bytes / max(1, quant_bytes)),
        "physical_blocks": float(cache.block_table.ge(0).sum().item()),
        "materializes_dense_kv": 0.0,
        "block_n": float(block_n),
        "num_warps": float(num_warps),
    }
    return out, stats


def paged_quant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 2 的 Triton 兼容入口：paged quant KV attention。

    当前行为：
    - 如果安装了 Triton，且 q/cache 都位于 CUDA，则启动真正的 paged Triton kernel。
    - 否则委托给 PyTorch 参考实现 paged_quant_attention，保证 CPU 环境可用。

    Triton 主路径：
    - kernel 内部直接读取 block_table，按 logical block 间接寻址 physical KV block。
    - 直接读取 INT8 K/V block。
    - 融合 INT8 反量化、QK、softmax 和 PV，避免物化 dense KV。
    """

    if (
        HAS_TRITON
        and _args
        and isinstance(_args[0], torch.Tensor)
        and _args[0].device.type == "cuda"
        and len(_args) >= 2
        and isinstance(_args[1], PagedKVCache)
    ):
        return _run_paged_quant_attention_triton(*_args, **_kwargs)

    # 非 CUDA 或未安装 Triton 时使用 portable fallback；这个 fallback 会调用 cache.to_dense()。
    from .attention import paged_quant_attention

    return paged_quant_attention(*_args, **_kwargs)
