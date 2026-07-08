from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import torch

from .attention import paged_quant_attention
from .paged import PagedKVCache
from .runtime import device_name, resolve_device, resolve_dtype, synchronize


@dataclass(frozen=True)
class ServingConfig:
    """decode serving 模拟的配置。

    字段说明：
    - requests: 并发请求数，也就是 batch size。
    - prompt_len: 每个请求初始 prompt 的 token 数。
    - decode_steps: 模拟逐 token decode 的步数。
    - heads: attention head 数量。
    - head_dim: 每个 attention head 的特征维度。
    - block_size: paged KV cache 中每个 block 容纳的 token 数。
    - seed: 随机数种子，用于让 benchmark 输入可复现。
    - device: 运行设备；"auto" 会由 runtime.resolve_device 自动选择。
    - dtype: 张量数据类型；"auto" 会由 runtime.resolve_dtype 根据设备选择。
    """

    # 并发请求数量；在 attention 中对应 batch 维度。
    requests: int = 8

    # 初始 prompt 长度，用来构造第一版 KV cache。
    prompt_len: int = 128

    # decode 阶段要生成多少个 token step。
    decode_steps: int = 16

    # attention head 数。
    heads: int = 8

    # 单个 attention head 的隐藏维度。
    head_dim: int = 64

    # paged cache 的 block 粒度；越大则 block_table 越短，但尾部 padding 可能越多。
    block_size: int = 16

    # 固定随机种子，保证每次模拟生成相同的伪输入。
    seed: int = 0

    # 运行设备配置，比如 "auto"、"cpu"、"cuda"。
    device: str = "auto"

    # 运行 dtype 配置，比如 "auto"、"float16"、"bfloat16"、"float32"。
    dtype: str = "auto"


def run_decode_simulation(config: ServingConfig) -> Dict[str, float]:
    """在 paged KV 路径上运行一个确定性的 decode loop 模拟。

    这个函数不接入真实 tokenizer 或模型层，而是用随机张量模拟 LLM serving
    中和 KV cache 相关的关键路径：
    1. 根据 prompt 构造初始 dense K/V。
    2. 将 dense K/V 转成 PagedKVCache，模拟 prefill 后的 KV cache。
    3. 执行一次 paged_quant_attention，统计近似 TTFT。
    4. 循环执行 decode attention，并把每一步生成的 next K/V append 到 cache。

    返回的字典字段：
    - device/device_name: 实际运行设备及设备名称。
    - dtype: 实际使用的数据类型。
    - requests/prompt_len/decode_steps: 本次模拟规模。
    - ttft_ms: first token 路径耗时，包含初始 paged cache 构建和第一次 attention。
    - tpot_ms: 每个 decode step 的平均耗时。
    - tokens_per_second: 按 requests * decode_steps 估算的 decode 吞吐。
    """

    # 解析运行设备；auto 会优先使用可用加速设备，否则回退 CPU。
    device = resolve_device(config.device)

    # 解析 dtype；auto 会结合 device 选择合适的默认精度。
    dtype = resolve_dtype(config.dtype, device)

    # CUDA 上允许 TF32 matmul，用于更接近常见推理 benchmark 配置。
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    # 使用指定 device 上的随机数生成器，并固定 seed，保证 benchmark 可复现。
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    # 初始 prompt KV 的 dense 形状：
    # [requests, heads, prompt_len, head_dim]。
    shape = (config.requests, config.heads, config.prompt_len, config.head_dim)

    # 用随机张量模拟 prefill 阶段产生的历史 K/V cache。
    k = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # 计时前同步设备，避免 CUDA 异步执行把上一步开销混入 TTFT。
    synchronize(device)
    start = perf_counter()

    # 把 dense prompt KV 转成 paged INT8 KV cache。
    # 这一步模拟 prefill 完成后服务端持有的压缩分页 KV cache。
    cache = PagedKVCache.from_dense(k, v, block_size=config.block_size)

    # 当前 decode token 的 query，形状为 [requests, heads, head_dim]。
    q = torch.randn((config.requests, config.heads, config.head_dim), generator=generator, device=device, dtype=dtype)

    # 执行第一次 decode attention，用于统计 first token 路径。
    _ = paged_quant_attention(q, cache)

    # 计时后再次同步，确保异步设备上的 attention 已完成。
    synchronize(device)

    # TTFT: time to first token，单位毫秒。
    ttft_ms = (perf_counter() - start) * 1000.0

    # decode 阶段单独计时，用来统计 TPOT 和吞吐。
    synchronize(device)
    decode_start = perf_counter()
    for _step in range(config.decode_steps):
        # 每一步生成一个新的 query token。
        q = torch.randn((config.requests, config.heads, config.head_dim), generator=generator, device=device, dtype=dtype)

        # 使用当前 paged KV cache 执行 decode attention。
        # out 形状为 [requests, heads, head_dim]。
        out = paged_quant_attention(q, cache)

        # 用 attention 输出构造下一步要追加进 cache 的 K。
        # unsqueeze(2) 插入 token 维度，变成 [requests, heads, 1, head_dim]。
        next_k = out.unsqueeze(2).to(k.dtype)

        # 用 tanh(out) 构造下一步 V，避免 K/V 完全相同；这仍是模拟数据，
        # 不代表真实模型里的 projection 逻辑。
        next_v = torch.tanh(out).unsqueeze(2).to(v.dtype)

        # 将新 token 的 K/V 追加到 paged cache，模拟自回归 decode 的 KV 增长。
        cache = cache.append(next_k, next_v)

    # 等待 decode loop 中所有异步 kernel 完成后停止计时。
    synchronize(device)

    # decode 阶段总耗时，单位毫秒。
    decode_ms = (perf_counter() - decode_start) * 1000.0

    # 总生成 token 数；每个 step 为每个 request 生成 1 个 token。
    generated = config.requests * config.decode_steps

    # 返回 benchmark 指标。数值统一转成 JSON/打印友好的基础类型。
    return {
        "device": str(device),
        "device_name": device_name(device),
        "dtype": str(dtype).replace("torch.", ""),
        "requests": float(config.requests),
        "prompt_len": float(config.prompt_len),
        "decode_steps": float(config.decode_steps),
        "ttft_ms": ttft_ms,
        "tpot_ms": decode_ms / max(1, config.decode_steps),
        "tokens_per_second": generated / max(1.0e-9, decode_ms / 1000.0),
    }
