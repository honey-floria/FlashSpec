"""benchmarks/ 和 scripts/ 共用的命令行小工具。

集中三类此前散落在各 CLI 里的重复逻辑：
- 逗号分隔参数解析（int / str 列表）。
- --dtype 的合法取值集合（原本在三个 argparse 里逐字复制）。
- 结果输出尾巴（--json 打印 JSON，否则逐行 key: value）。
- microbench 子进程参数列表构建（ncu 回填和 profiling 命令模板复用同一份）。
"""

from __future__ import annotations

import json
from typing import Any, Iterable


# --dtype 的统一合法取值；microbench / sweep / e2e_serving 共用，避免各写一份。
DTYPE_CHOICES = ["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"]


def parse_int_list(value: str) -> list[int]:
    """把逗号分隔字符串解析成 int 列表，忽略空项。"""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    """把逗号分隔字符串解析成去空白的 str 列表，忽略空项。"""

    return [item.strip() for item in value.split(",") if item.strip()]


def emit_result(result: dict[str, Any], *, as_json: bool) -> None:
    """把结果字典打印到 stdout：as_json 时输出缩进 JSON，否则逐行 key: value。"""

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


def microbench_cli_args(
    *,
    backend: str,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int,
    iters: int,
    warmup: int,
    repeats: int,
    device: str,
    dtype: str,
    paged_layout: str,
    layout_seed: int,
    length_pattern: str,
    seed: int,
    lengths: str = "",
    include_json_flag: bool = True,
) -> list[str]:
    """构建复现一次 microbench 运行所需的 CLI 参数列表（不含前导 python/ncu）。

    ncu 回填子进程和 JSON 里的 profiling 命令模板都复用这一份，避免 shape/timing/
    layout 参数在多处重复拼装、错位。调用方在前面拼上 python 解释器和脚本路径即可。
    """

    args = [
        "--backend", backend,
        "--batch", str(batch),
        "--heads", str(heads),
        "--seq-len", str(seq_len),
        "--head-dim", str(head_dim),
        "--block-size", str(block_size),
        "--iters", str(max(1, iters)),
        "--warmup", str(max(1, warmup)),
        "--repeats", str(max(1, repeats)),
        "--device", device,
        "--dtype", dtype,
        "--paged-layout", paged_layout,
        "--layout-seed", str(layout_seed),
        "--length-pattern", length_pattern,
        "--seed", str(seed),
    ]
    if include_json_flag:
        args.append("--json")
    if lengths:
        args += ["--lengths", lengths]
    return args
