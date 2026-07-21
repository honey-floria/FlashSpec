from __future__ import annotations

import os


# Triton 是可选依赖：
# - 安装了 `.[triton]` 时，可以使用真实 Triton kernel。
# - 未安装 Triton 时，项目仍然应该能在 CPU/普通 PyTorch 环境中导入和测试。
try:
    # triton: Triton Python API，用于定义 @triton.jit kernel 和 launch。
    import triton  # type: ignore

    # tl: Triton language namespace，kernel 中使用 tl.load/tl.store/tl.exp 等。
    import triton.language as tl  # type: ignore

    # HAS_TRITON 标记当前环境是否可以使用 Triton 后端。
    HAS_TRITON = True
except ModuleNotFoundError:
    # 没有安装 Triton 时，保留同名变量，避免其他模块引用时报 NameError。
    triton = None
    tl = None

    # 当前环境只能使用 portable PyTorch fallback。
    HAS_TRITON = False


def next_power_of_2(x: int) -> int:
    """返回不小于 x 的最小 2 的幂，用作 Triton block_d。"""

    return 1 << (x - 1).bit_length()


def validate_decode_query(q, *, kernel_name: str) -> None:
    """两个 Triton kernel 共用的 query 前置校验：ndim / CUDA / head_dim 上限。

    只检查 decode query 本身的通用约束，K/V 对齐、量化布局等 kernel 专有约束
    仍由各自的 _validate_* 负责。kernel_name 用于拼出可读的错误信息。
    """

    if q.ndim != 3:
        raise ValueError("q 必须是 [batch, heads, head_dim] 形状")
    if q.device.type != "cuda":
        raise RuntimeError(f"Triton {kernel_name} attention 需要 CUDA q tensor")
    if q.shape[-1] > 256:
        raise ValueError(f"Triton {kernel_name} attention 当前只支持 head_dim <= 256")


def dispatch_triton_or_fallback(args, kwargs, *, run_triton, load_fallback, extra_guard=None):
    """两个 kernel 兼容入口共用的 CUDA-gating 分发逻辑。

    当且仅当安装了 Triton、第一个位置参数是 CUDA tensor、且可选的 extra_guard
    也通过时，走真正的 Triton kernel（run_triton）；否则调用 load_fallback()
    延迟拿到 PyTorch 参考实现再执行。延迟导入是为了避免与 attention 模块循环依赖。

    参数：
    - args/kwargs: 透传给 run_triton 或 fallback 的原始调用参数。
    - run_triton: 真正的 Triton launcher。
    - load_fallback: 返回 PyTorch fallback 函数的零参可调用（内部做延迟 import）。
    - extra_guard: 可选谓词，接收 args，返回是否满足额外的 Triton 走通条件。
    """

    # torch 只在这里按需导入，保持 triton_utils 顶层不强依赖 torch。
    import torch

    if (
        HAS_TRITON
        and args
        and isinstance(args[0], torch.Tensor)
        and args[0].device.type == "cuda"
        and (extra_guard is None or extra_guard(args))
    ):
        return run_triton(*args, **kwargs)
    return load_fallback()(*args, **kwargs)


# 两个 Triton kernel（fused / paged）共享的默认 tile / warp 配置。
# 每轮扫描的 token 数：block_n 越大寄存器压力越高、占用率天花板越低。
_DEFAULT_BLOCK_N = 128


def resolve_block_n() -> int:
    """每个 program 每轮扫描的 token 数。默认 128，可用 FLASHSPEC_BLOCK_N 覆盖。

    profiling matrix 结论：block_n=128 的有效 DRAM throughput 最好；
    block_n=32 虽然 occupancy 更高但明显更慢。保留 A/B 开关：调小 block_n
    可降低 registers_per_thread、抬高占用率天花板，代价是循环轮数增加。
    仅接受 [16, 128] 内的 2 的幂，避免病态 tile 尺寸。
    """

    override = os.environ.get("FLASHSPEC_BLOCK_N")
    if override is not None:
        try:
            v = int(override)
        except ValueError:
            v = 0
        if v in (16, 32, 64, 128):
            return v
    return _DEFAULT_BLOCK_N


def resolve_num_warps() -> int:
    """Triton program 的 warp 数。默认 4，可用 FLASHSPEC_NUM_WARPS 覆盖做 profiling sweep。

    早期 A100 实测显示 8 warp 比 4 warp 慢，但保留开关便于结合 block_n/Split-K
    做矩阵验证，避免每次实验都改代码。仅接受 1/2/4/8。
    """

    override = os.environ.get("FLASHSPEC_NUM_WARPS")
    if override is not None:
        try:
            v = int(override)
        except ValueError:
            v = 0
        if v in (1, 2, 4, 8):
            return v
    return 4


# ---------------------------------------------------------------------------
# 两个 decode attention kernel（fused / paged）共用的 @triton.jit 小工具。
#
# 这些 helper 只做与 KV 布局无关的通用步骤：program-id 拆分、head_dim 偏移、
# query 载入、有效长度、softmax 归一。真正区分 fused（连续 KV）和 paged
# （block_table 间接寻址）的 tile 级地址计算仍保留在各自的 kernel 里。
#
# 跨模块调用 @triton.jit：被调用的 jit 函数通过调用方所在模块的 globals 解析，
# 因此 triton_fused / triton_paged 只要 import 下面这些名字即可在 kernel 内调用。
# 注意：本段只有在 Triton 可用时才定义；CPU 环境下这些名字不存在，但那时
# 也不会有 kernel 去调用它们（走 PyTorch fallback）。
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def pid_to_batch_head(pid, heads: tl.constexpr):
        """把展平后的 program id 还原成 batch/head 下标。"""

        batch_idx = pid // heads
        head_idx = pid - batch_idx * heads
        return batch_idx, head_idx

    @triton.jit
    def make_dim_offsets(head_dim: tl.constexpr, block_d: tl.constexpr):
        """生成 head_dim 维度上的向量化偏移和 padding mask。"""

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        return offs_d, d_mask

    @triton.jit
    def load_query(
        q_ptr,
        batch_idx,
        head_idx,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        offs_d,
        d_mask,
    ):
        """读取当前 [batch, head] 的 query 向量，并返回它的展平 base 地址。"""

        q_base = (batch_idx * heads + head_idx) * head_dim
        q = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        return q_base, q

    @triton.jit
    def effective_length(
        lengths_ptr,
        batch_idx,
        seq_len: tl.constexpr,
        has_lengths: tl.constexpr,
    ):
        """返回当前 batch 的有效 sequence 长度（min(lengths[b], seq_len)）。"""

        effective_len = seq_len
        if has_lengths:
            loaded_len = tl.load(lengths_ptr + batch_idx)
            effective_len = tl.minimum(loaded_len, seq_len)
        return effective_len

    @triton.jit
    def normalize_acc(acc, l):
        """将未归一化 PV 累积除以 softmax 分母；空序列输出 0。"""

        denom = tl.where(l > 0.0, l, 1.0)
        out = acc / denom
        return tl.where(l > 0.0, out, 0.0)

else:  # pragma: no cover - CPU 环境无 Triton
    # CPU 环境不编译任何 kernel，这些 helper 永远不会被调用。但 triton_fused /
    # triton_paged 在模块顶层无条件 import 这些名字，所以必须始终可导入：给出
    # 会主动报错的占位符，避免误在非 Triton 环境下调用时静默出错。
    def _triton_only(*_args, **_kwargs):
        raise RuntimeError("该 Triton jit helper 仅在安装 Triton 且运行于 CUDA 时可用")

    pid_to_batch_head = _triton_only
    make_dim_offsets = _triton_only
    load_query = _triton_only
    effective_length = _triton_only
    normalize_acc = _triton_only
