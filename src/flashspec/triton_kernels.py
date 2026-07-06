from __future__ import annotations

from typing import Any


try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
except ModuleNotFoundError:
    triton = None
    tl = None
    HAS_TRITON = False


def fused_dequant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 1 compatible entry point.

    A CUDA deployment can replace this wrapper with a custom Triton launch. The
    repository keeps the public signature usable on CPU by delegating to the
    verified portable backend.
    """

    from .attention import fused_dequant_attention

    return fused_dequant_attention(*_args, **_kwargs)


def paged_quant_attention_triton(*_args: Any, **_kwargs: Any) -> Any:
    """Kernel 2 compatible entry point."""

    from .attention import paged_quant_attention

    return paged_quant_attention(*_args, **_kwargs)
