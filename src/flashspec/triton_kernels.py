from __future__ import annotations

from typing import Any


# Triton 是可选依赖：
# - 安装了 `.[triton]` 时，可以在这里接入真正的 Triton kernel。
# - 未安装 Triton 时，项目仍然应该能在 CPU/普通 PyTorch 环境中导入和测试。
try:
    # triton: Triton Python API，通常用于定义 @triton.jit kernel 和 launch。
    import triton  # type: ignore

    # tl: Triton language namespace，真实 kernel 中会用 tl.load/tl.store/tl.dot 等。
    import triton.language as tl  # type: ignore

    # HAS_TRITON 标记当前环境是否可以使用 Triton 后端。
    HAS_TRITON = True
except ModuleNotFoundError:
    # 没有安装 Triton 时，保留同名变量，避免其他模块引用时报 NameError。
    triton = None
    tl = None

    # 当前环境只能使用 portable PyTorch fallback。
    HAS_TRITON = False


def fused_dequant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 1 的 Triton 兼容入口：fused dequant attention。

    参数：
    - *_args/**_kwargs: 透传给 attention.fused_dequant_attention 的所有参数。
      当前 wrapper 不重新声明完整签名，是为了和 portable backend 保持兼容，
      同时给未来 Triton kernel 留出替换空间。

    当前行为：
    - 不直接启动 Triton kernel。
    - 延迟导入并调用 PyTorch 参考实现 fused_dequant_attention。

    设计意图：
    - 让外部代码可以先依赖 `fused_dequant_attention_triton` 这个稳定入口。
    - 后续 CUDA/Triton 部署可以在这里替换成真正的 fused kernel launch。
    - 即使没有安装 Triton，CPU 测试和基础功能也能正常跑通。
    """

    # 延迟导入可以避免模块加载时产生循环依赖，也让没有 Triton 的环境仍可导入本文件。
    from .attention import fused_dequant_attention

    # 当前 fallback 直接委托给已测试过的 portable PyTorch 后端。
    return fused_dequant_attention(*_args, **_kwargs)


def paged_quant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 2 的 Triton 兼容入口：paged quant KV attention。

    参数：
    - *_args/**_kwargs: 透传给 attention.paged_quant_attention 的所有参数。

    当前行为：
    - 不直接根据 block_table 发起 Triton kernel。
    - 委托给 PyTorch 参考实现 paged_quant_attention，该实现会先
      cache.to_dense()，再执行 reference attention。

    未来替换方向：
    - 在这里接入真正的 paged attention Triton launch。
    - kernel 内部直接读取 block_table，按 logical block 间接寻址
      physical KV block。
    - 同时融合 INT8 反量化和 attention 计算，避免物化 dense KV。
    """

    # 延迟导入 portable backend，保持模块导入轻量且避免循环依赖。
    from .attention import paged_quant_attention

    # 当前 fallback 直接复用 PyTorch 参考实现，保证语义和测试结果一致。
    return paged_quant_attention(*_args, **_kwargs)
