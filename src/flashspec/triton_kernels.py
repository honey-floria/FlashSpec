from __future__ import annotations

"""Triton kernel 兼容导出层。

历史上 Kernel 1、Kernel 2、Triton 可选导入和 Python wrapper 都放在这个文件里。
现在真实实现已按功能拆分到：

- `triton_utils.py`：Triton 可选导入、`HAS_TRITON` 和共享工具。
- `triton_fused.py`：Kernel 1，INT8 KV fused dequant attention。
- `triton_paged.py`：Kernel 2，paged INT8 KV attention。

保留本文件是为了兼容现有导入路径，例如：
`from flashspec.triton_kernels import HAS_TRITON`。
"""

from .triton_fused import fused_dequant_attention_triton
from .triton_paged import paged_quant_attention_triton
from .triton_utils import HAS_TRITON, tl, triton

__all__ = [
    "HAS_TRITON",
    "fused_dequant_attention_triton",
    "paged_quant_attention_triton",
    "tl",
    "triton",
]
