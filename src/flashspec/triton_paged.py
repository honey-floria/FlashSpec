"""Triton paged INT8 KV attention.

本文件实现 Kernel 2：K/V cache 不再按 [batch, heads, seq, dim] 连续存放，
而是拆成 physical blocks。每条 request 通过 block_table 把自己的 logical block
映射到实际 physical block。kernel 在 Triton 内部完成 block_table 间接寻址、
INT8 反量化和 decode attention，不调用 cache.to_dense()。

整体数据流：
1. Python launcher 接收 q 和 PagedKVCache。
2. cache.block_table[b, logical_block] 给出 batch b 的 logical block 对应的 physical block。
3. kernel 按逻辑 token 位置扫描，先算 logical_block/page_offset。
4. 通过 block_table 读 physical_block，再定位到 physical K/V block store 中的 INT8 元素。
5. 用 scale/zero_point 反量化 K/V，在线完成 softmax(QK) @ V。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import torch

from .paged import PagedKVCache
from .quant import build_compression_stats
from .triton_utils import (
    HAS_TRITON,
    dispatch_triton_or_fallback,
    effective_length,
    load_query,
    make_dim_offsets,
    next_power_of_2,
    normalize_acc,
    pid_to_batch_head,
    resolve_block_n as _resolve_block_n,
    resolve_num_warps as _resolve_num_warps,
    tl,
    triton,
    validate_decode_query,
)


@dataclass(frozen=True)
class PagedKVBuffers:
    """paged kernel 的 query / INT8 KV / block_table / lengths 输入指针。

    把 9 个 tensor 参数收拢成一个对象，避免 launcher 里逐个平铺、错位传参。
    字段顺序与 kernel signature 的前 9 个位置参数严格对应。与 fused 相比多出
    block_table 一项，用于 logical block -> physical block 的间接寻址。
    """

    q: torch.Tensor
    k_values: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_values: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    block_table: torch.Tensor
    lengths: torch.Tensor

    def as_args(self) -> tuple:
        """按 kernel signature 顺序返回输入 tensor 的位置参数元组。"""

        return (
            self.q,
            self.k_values,
            self.k_scale,
            self.k_zero,
            self.v_values,
            self.v_scale,
            self.v_zero,
            self.block_table,
            self.lengths,
        )


@dataclass(frozen=True)
class PagedAttentionMeta:
    """paged kernel 的 constexpr 元参数与 launch 配置。

    这些值决定 kernel 的 block_table 间接寻址、量化 block 定位、softmax 缩放、
    tile 尺寸和 warp 数。core_args() 按 kernel signature 顺序给出全部 constexpr，
    num_warps 由 launcher 作为关键字参数单独传入。
    """

    heads: int
    max_seq_len: int
    head_dim: int
    table_cols: int
    physical_quant_blocks: int
    page_block_size: int
    quant_block_size: int
    sm_scale: float
    block_n: int
    block_d: int
    num_warps: int

    def core_args(self) -> tuple:
        """返回 kernel 的全部 constexpr 位置参数（不含 num_warps 关键字参数）。"""

        return (
            self.heads,
            self.max_seq_len,
            self.head_dim,
            self.table_cols,
            self.physical_quant_blocks,
            self.page_block_size,
            self.quant_block_size,
            self.sm_scale,
            self.block_n,
            self.block_d,
        )


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

        函数作用：
        - 这是 paged cache 的主 Triton kernel。
        - grid 只有一维，大小为 batch * heads。
        - 每个 program 计算一个 [batch, head] 的 decode attention 输出。

        参数含义：
        - q_ptr: 输入 query，形状 [batch, heads, head_dim]。
        - k_values_ptr/v_values_ptr: 物理 block store 中的 INT8 K/V values，逻辑形状
          [physical_blocks, heads, physical_quant_blocks, quant_block_size, head_dim]。
        - k_scale_ptr/v_scale_ptr: K/V 每个 physical quant block 的 scale，逻辑形状
          [physical_blocks, heads, physical_quant_blocks, 1, 1]。
        - k_zero_point_ptr/v_zero_point_ptr: 与 scale 同布局的 zero_point，位于 uint8 域。
        - block_table_ptr: [batch, table_cols]，logical block -> physical block 的映射表。
        - lengths_ptr: [batch]，每条 request 的真实有效 token 数。
        - out_ptr: 输出 tensor，形状 [batch, heads, head_dim]。
        - heads: attention head 数。
        - max_seq_len: 当前 batch 中 kernel 需要扫描的最大逻辑 sequence 长度。
        - head_dim: 每个 head 的真实维度。
        - table_cols: block_table 的列数，即每条 request 最多有多少 logical block。
        - physical_quant_blocks: 每个 physical page 内含多少个量化 block。
        - page_block_size: paged cache 的逻辑/物理 page token 数。
        - quant_block_size: INT8 量化 block 的 token 数。
        - sm_scale: softmax scale，通常是 1 / sqrt(head_dim)。
        - block_n: 每轮扫描的逻辑 token tile 大小。
        - block_d: head_dim 向上取 2 的幂后的 tile 大小。

        数据逻辑：
        - logical token 位置 -> logical_block/page_offset。
        - block_table[batch, logical_block] -> physical_block。
        - page_offset -> physical_quant_block/quant_offset。
        - physical_block/head/physical_quant_block/quant_offset/dim -> INT8 K/V 地址。

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

        # pid 对应展平后的 [batch, head] 编号。以下 4 步（pid 拆分、head_dim 偏移、
        # query 载入、有效长度）与 Kernel 1 完全一致，复用 triton_utils 的共享 jit helper。
        pid = tl.program_id(0)
        batch_idx, head_idx = pid_to_batch_head(pid, heads)

        # offs_d 长度为 block_d，可能大于真实 head_dim；d_mask 屏蔽 padding lane。
        offs_d, d_mask = make_dim_offsets(head_dim, block_d)

        # q_base 是当前 [batch, head] query 向量在展平内存（[batch, heads, head_dim]）中的起始下标。
        q_base, q = load_query(q_ptr, batch_idx, head_idx, heads, head_dim, offs_d, d_mask)

        # paged cache 恒带 lengths，用 max_seq_len 作为编译期扫描上界。
        effective_len = effective_length(lengths_ptr, batch_idx, max_seq_len, True)

        # online softmax 状态，避免保存完整 scores。
        # m: 已扫描 scores 的最大值。
        # l: exp(scores - m) 的累积和。
        # acc: 未归一化的 softmax(scores) @ V 累积。
        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)

        # 按逻辑 sequence 位置扫描；每个 token 再通过 block_table 映射到物理 block。
        for start in range(0, max_seq_len, block_n):
            # offs_s: 当前 tile 的逻辑 token 下标，形状 [block_n]。
            # s_mask: token 是否位于当前 request 的有效长度内。
            offs_s = start + tl.arange(0, block_n)
            s_mask = offs_s < effective_len

            # logical_block 是 token 属于该 request 的第几个逻辑页。
            # page_offset 是 token 在该页内的位置。
            logical_block = offs_s // page_block_size
            page_offset = offs_s - logical_block * page_block_size

            # block_table 布局是 [batch, table_cols]。
            # 未分配的 logical block 通常是 -1；有效长度 mask 会避免访问无效 token。
            # table_mask 同时保证 logical_block 没有越过 block_table 的列数。
            table_mask = s_mask & (logical_block < table_cols)
            physical_block = tl.load(
                block_table_ptr + batch_idx * table_cols + logical_block,
                mask=table_mask,
                other=0,
            )
            # physical_valid 表示这个 token 对应到一个真实分配的 physical block。
            physical_valid = table_mask & (physical_block >= 0)

            # physical block 内部的 sequence 位置还要映射到量化 block。
            # 当前主路径 page_block_size == quant_block_size，避免在内层循环做
            # 恒为 0 的除法/乘法地址计算；非等长配置保留通用路径。
            if page_block_size == quant_block_size:
                # 一个 page 正好是一个量化 block 时，page 内 quant block 编号恒为 0。
                physical_quant_block = tl.full((block_n,), 0, tl.int64)
                quant_offset = page_offset
                qparam_mask = physical_valid
            else:
                # 一个 page 内有多个量化 block 时，需继续按 quant_block_size 分块。
                physical_quant_block = page_offset // quant_block_size
                quant_offset = page_offset - physical_quant_block * quant_block_size
                qparam_mask = physical_valid & (physical_quant_block < physical_quant_blocks)

            # QuantizedTensor.values 物理布局：
            # [physical_blocks, heads, physical_quant_blocks, quant_block_size, head_dim]。
            # kv_offsets 是 [block_n, block_d] 地址矩阵，对应当前逻辑 token tile 映射到的物理 K/V 元素。
            kv_offsets = (
                ((((physical_block[:, None] * heads + head_idx) * physical_quant_blocks + physical_quant_block[:, None])
                  * quant_block_size + quant_offset[:, None])
                 * head_dim)
                + offs_d[None, :]
            )
            # kv_mask 屏蔽无效 physical block、越界量化 block 和 head_dim padding 列。
            kv_mask = qparam_mask[:, None] & d_mask[None, :]

            # scale/zero_point 布局是 [physical_blocks, heads, physical_quant_blocks, 1, 1]。
            # qparam_offsets 是 [block_n]，每个 token 使用所在 quant block 的量化参数。
            qparam_offsets = (physical_block * heads + head_idx) * physical_quant_blocks + physical_quant_block
            k_scale = tl.load(k_scale_ptr + qparam_offsets, mask=qparam_mask, other=1.0).to(tl.float32)
            k_zero = tl.load(k_zero_point_ptr + qparam_offsets, mask=qparam_mask, other=0).to(tl.float32)

            # INT8 values 以 signed int8 存储，反量化时先 +128 回到 uint8 域。
            # k_deq 形状 [block_n, block_d]，只在寄存器中存在，不写回全局内存。
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
            # v_deq 形状 [block_n, block_d]；与 probs 相乘后沿 token 维累积到 acc。
            v_deq = (v_i8 + 128.0 - v_zero[:, None]) * v_scale[:, None]

            # 累积 PV。acc 始终和当前 new_m 对齐。
            acc = acc * old_scale + tl.sum(probs[:, None] * v_deq, axis=0)
            m = new_m
            l = new_l

        # 空长度 request 输出 0，避免除零（与 Kernel 1 共用 normalize_acc）。
        out = normalize_acc(acc, l)

        # 输出布局是 [batch, heads, head_dim]。
        tl.store(out_ptr + q_base + offs_d, out, mask=d_mask)


def _validate_triton_paged_inputs(q: torch.Tensor, cache: PagedKVCache) -> None:
    """校验 Triton paged attention 的输入约束。

    函数作用：
    - 在 Python 侧检查 q 和 PagedKVCache 的 shape/device 约束。
    - 避免错误的 block_table、lengths 或量化布局进入 kernel 后产生非法访问。

    参数含义：
    - q: decode query，形状 [batch, heads, head_dim]。
    - cache: PagedKVCache，包含物理 K/V block store、block_table、lengths 和 block_size。
    Kernel 2 当前实现聚焦 paged decode attention 主路径：
    - q 必须是 [batch, heads, head_dim]；
    - cache 内的 K/V 必须是 PagedKVCache 的物理 block 布局；
    - q、量化 K/V、block_table、lengths 必须位于同一个 CUDA 设备；
    - head_dim 当前限制为不超过 256，覆盖常见 64/128 配置。
    """

    validate_decode_query(q, kernel_name="paged")
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
    if cache.k_quant.device != q.device or cache.v_quant.device != q.device:
        raise RuntimeError("q 和 cache 的量化 K/V 必须位于同一个 CUDA 设备")
    if cache.block_table.device != q.device or cache.lengths.device != q.device:
        raise RuntimeError("block_table、lengths 必须与 q 位于同一个 CUDA 设备")


def _run_paged_quant_attention_triton(
    q: torch.Tensor,
    cache: PagedKVCache,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """运行真正的 Triton Kernel 2，不调用 cache.to_dense()。

    函数作用：
    - 这是 Python 层的 paged Triton launcher。
    - 负责输入校验、contiguous 视图整理、元参数计算和 kernel launch。
    - 只传递 block_table 与物理 INT8 block store 指针，不构造 dense K/V。

    参数含义：
    - q: [batch, heads, head_dim] 的 query。
    - cache: PagedKVCache，内部 K/V 已按 physical block 量化。
    - return_stats: 是否返回 benchmark/report 用统计字段。

    返回：
    - return_stats=False: out，形状 [batch, heads, head_dim]。
    - return_stats=True: (out, stats)，stats 包含压缩率、物理 block 数、kernel 参数等。

    数据逻辑：
    - block_table 和 lengths 是唯一的 per-request 元数据。
    - kernel 根据 logical token 动态查 physical block，因此支持非连续、打乱或交错的 cache 布局。
    - attention 的数学结果应与先 cache.to_dense() 再做 reference attention 等价。

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
    q_contig = q.contiguous()  # [batch, heads, head_dim]。
    k_values = cache.k_quant.values.contiguous()  # [physical_blocks, heads, physical_quant_blocks, quant_block_size, head_dim]。
    k_scale = cache.k_quant.scale.contiguous()  # [physical_blocks, heads, physical_quant_blocks, 1, 1]。
    k_zero = cache.k_quant.zero_point.contiguous()  # 与 scale 同布局，保存 uint8 域 zero_point。
    v_values = cache.v_quant.values.contiguous()
    v_scale = cache.v_quant.scale.contiguous()
    v_zero = cache.v_quant.zero_point.contiguous()
    block_table = cache.block_table.contiguous()  # [batch, table_cols]，logical block -> physical block。
    lengths = cache.lengths.contiguous()  # [batch]，每条 request 的有效 token 数。

    batch, heads, head_dim = q_contig.shape  # request 数 / attention head 数 / 单 head 维度。
    max_seq_len = int(cache.max_seq_len)  # kernel 的逻辑扫描上界。
    table_cols = int(block_table.shape[1])  # block_table 中 logical block 列数。
    physical_quant_blocks = int(k_values.shape[-3])  # 一个 physical page 内包含的量化 block 数。
    page_block_size = int(cache.block_size)  # paged cache 的 page token 数。
    quant_block_size = int(cache.k_quant.block_size)  # 量化时每个 scale/zero_point 覆盖的 token 数。
    block_d = next_power_of_2(int(head_dim))  # Triton feature tile，head_dim 向上取 2 的幂。

    # 和 Kernel 1 保持一致，默认每次扫描 128 个逻辑 token；profiling 可用
    # FLASHSPEC_BLOCK_N 覆盖，用于观察 block_table 间接寻址下的 tile 取舍。
    block_n = _resolve_block_n()
    num_warps = _resolve_num_warps()
    out = torch.empty_like(q_contig)  # 与 q 同形状同 dtype；kernel 内部用 FP32 累积后写回。

    # 把 9 个输入指针和 constexpr 元参数收拢成两个对象，按 kernel signature 固定
    # 顺序组装 launch，避免逐个平铺、错位传参（与 fused launcher 同一模式）。
    buffers = PagedKVBuffers(
        q=q_contig,
        k_values=k_values,
        k_scale=k_scale,
        k_zero=k_zero,
        v_values=v_values,
        v_scale=v_scale,
        v_zero=v_zero,
        block_table=block_table,
        lengths=lengths,
    )
    meta = PagedAttentionMeta(
        heads=heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        table_cols=table_cols,
        physical_quant_blocks=physical_quant_blocks,
        page_block_size=page_block_size,
        quant_block_size=quant_block_size,
        sm_scale=1.0 / math.sqrt(float(head_dim)),
        block_n=block_n,
        block_d=block_d,
        num_warps=num_warps,
    )

    grid = (batch * heads,)  # 一个 program 处理一个 [batch, head]。
    _paged_quant_attention_kernel[grid](
        *buffers.as_args(),
        out,
        *meta.core_args(),
        num_warps=meta.num_warps,
    )

    if not return_stats:
        return out

    dense_bytes = 2 * batch * heads * max_seq_len * head_dim * q_contig.element_size()
    quant_bytes = cache.estimated_bytes()
    # 真正的 paged kernel 直接按 block_table 间接寻址，不物化 dense KV。
    stats = build_compression_stats(
        dense_bytes,
        quant_bytes,
        materializes_dense_kv=False,
        physical_blocks=float(cache.block_table.ge(0).sum().item()),
        block_n=float(block_n),
        num_warps=float(num_warps),
    )
    return out, stats


def paged_quant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 2 的 Triton 兼容入口：paged quant KV attention。

    函数作用：
    - 对外暴露统一入口，调用方不需要关心当前环境是否可运行 Triton。
    - CUDA + Triton + PagedKVCache 时走真正 paged kernel；否则走 PyTorch fallback。

    参数/返回：
    - 透传给 _run_paged_quant_attention_triton 或 attention.paged_quant_attention。
    - 返回形状与 q 相同的 attention 输出，或 return_stats=True 时返回 (out, stats)。

    当前行为：
    - 如果安装了 Triton，且 q/cache 都位于 CUDA，则启动真正的 paged Triton kernel。
    - 否则委托给 PyTorch 参考实现 paged_quant_attention，保证 CPU 环境可用。

    Triton 主路径：
    - kernel 内部直接读取 block_table，按 logical block 间接寻址 physical KV block。
    - 直接读取 INT8 K/V block。
    - 融合 INT8 反量化、QK、softmax 和 PV，避免物化 dense KV。
    """

    # paged 比 fused 多两个走通条件：至少要有 cache 参数、且它是 PagedKVCache。
    def _needs_paged_cache(args) -> bool:
        return len(args) >= 2 and isinstance(args[1], PagedKVCache)

    # 非 CUDA 或未安装 Triton 时的 portable fallback 会调用 cache.to_dense()。
    def _load_fallback():
        from .attention import paged_quant_attention

        return paged_quant_attention

    return dispatch_triton_or_fallback(
        _args,
        _kwargs,
        run_triton=_run_paged_quant_attention_triton,
        load_fallback=_load_fallback,
        extra_guard=_needs_paged_cache,
    )
