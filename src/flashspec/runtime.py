from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """解析用户传入的设备字符串，返回 torch.device。

    参数：
    - device: 设备配置字符串。
      - "auto": 自动选择设备；有 CUDA 时使用 cuda，否则使用 cpu。
      - "cpu": 明确使用 CPU。
      - "cuda" / "cuda:0" 等：明确使用 CUDA 设备。

    返回：
    - torch.device 对象，供后续张量创建、同步和设备名称查询使用。

    注意：
    - 如果用户明确请求 CUDA，但当前 PyTorch 检测不到 CUDA，会直接报错。
      这样可以避免 benchmark 静默回退到 CPU，导致性能数据被误读。
    """

    # auto 模式用于命令行 benchmark 的默认行为：
    # 有可用 GPU 就跑 CUDA，否则保持 CPU 可运行。
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将用户输入解析为标准 torch.device，例如 "cpu"、"cuda"、"cuda:0"。
    resolved = torch.device(device)

    # 用户显式指定 CUDA 时，必须确保当前环境真的可用 CUDA。
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 false。")
    return resolved


def resolve_dtype(dtype: str = "auto", device: torch.device | None = None) -> torch.dtype:
    """解析用户传入的 dtype 字符串，返回 torch.dtype。

    参数：
    - dtype: dtype 配置字符串。
      - "auto": 根据设备自动选择；CUDA 默认 float16，CPU 默认 float32。
      - "float16" / "fp16": torch.float16。
      - "bfloat16" / "bf16": torch.bfloat16。
      - "float32" / "fp32": torch.float32。
    - device: 可选设备信息，仅在 dtype="auto" 时用于决定默认 dtype。

    返回：
    - torch.dtype 对象，用于创建 benchmark 输入张量。

    设计意图：
    - CUDA 上默认 fp16，更贴近 LLM decode attention 的常见推理精度。
    - CPU 上默认 fp32，避免很多 CPU 算子对 fp16 支持较弱或速度异常。
    """

    # auto 模式：GPU 使用 float16，CPU 使用 float32。
    if dtype == "auto":
        return torch.float16 if device is not None and device.type == "cuda" else torch.float32

    # 支持长名字和常用缩写，方便命令行参数输入。
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    # 未知 dtype 直接报错，避免用错精度时 benchmark 继续运行。
    if dtype not in mapping:
        raise ValueError(f"不支持的 dtype: {dtype}")
    return mapping[dtype]


def synchronize(device: torch.device) -> None:
    """在需要时同步设备上的异步计算。

    参数：
    - device: 当前运行设备。

    CUDA kernel 默认是异步提交的，如果不在计时前后同步，
    perf_counter 记录到的可能只是 kernel launch 时间，而不是真实执行时间。

    CPU 执行通常是同步的，因此 CPU 路径不需要额外操作。
    """

    # 只有 CUDA 需要显式同步；CPU 分支保持 no-op。
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device: torch.device) -> str:
    """返回用于 benchmark/report 的设备名称。

    参数：
    - device: 当前运行设备。

    返回：
    - CUDA: 具体 GPU 名称，例如 NVIDIA 设备名。
    - CPU: 字符串 "cpu"。
    """

    # CUDA 返回具体显卡名称，便于比较不同硬件上的 benchmark 结果。
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)

    # 当前项目只区分 CUDA 和 CPU；非 CUDA 统一按 cpu 展示。
    return "cpu"
