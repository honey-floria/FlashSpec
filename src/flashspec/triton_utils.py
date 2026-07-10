from __future__ import annotations


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
