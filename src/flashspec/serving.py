from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Sequence

import torch

from .attention import paged_quant_attention
from .paged import PagedKVAllocator
from .runtime import device_name, resolve_device, resolve_dtype, synchronize


@dataclass(frozen=True)
class ServingConfig:
    """decode serving 模拟的配置。"""

    requests: int = 8  # 并发请求数量；在 attention 中对应 active batch size。
    prompt_len: int = 128  # 兼容旧 CLI 的单一 prompt 长度。
    prompt_lens: tuple[int, ...] | None = None  # 可选 prompt 长度候选列表；用于 variable-length 模拟。
    prompt_length_distribution: str = "uniform"  # prompt 长度采样方式：uniform/cycle、random、bimodal。
    decode_steps: int = 16  # decode 阶段要生成多少个全局 step。
    request_life_steps: int = 0  # 每个 request 生成多少 token 后结束并释放 block；0 表示不模拟 finish/arrival。
    heads: int = 8  # attention head 数。
    head_dim: int = 64  # 单个 attention head 的隐藏维度。
    block_size: int = 16  # paged cache 的 block 粒度。
    allocator_blocks: int | None = None  # allocator 的 physical block 容量；None 时按模拟规模自动估算。
    seed: int = 0  # 固定随机种子，保证每次模拟生成相同的伪输入。
    device: str = "auto"  # 运行设备配置，比如 "auto"、"cpu"、"cuda"。
    dtype: str = "auto"  # 运行 dtype 配置，比如 "auto"、"float16"、"bfloat16"、"float32"。


def run_decode_simulation(config: ServingConfig) -> Dict[str, float | int | str | list[int]]:
    """在 paged KV allocator 上运行一个确定性的 serving 模拟。

    模拟覆盖：
    1. prefill 阶段：为 active requests 构造 prompt KV 并分配 physical blocks。
    2. TTFT：统计 prefill 后第一次 decode attention。
    3. decode 阶段：逐 token attention + append，只更新被写入的 blocks。
    4. 可选 request finish/arrival：释放完成请求并复用 blocks 承接新请求。
    """

    _validate_config(config)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    prompt_lens = _resolve_prompt_lens(config)
    allocator = PagedKVAllocator(
        capacity_blocks=_estimate_allocator_blocks(config, prompt_lens),
        heads=config.heads,
        head_dim=config.head_dim,
        block_size=config.block_size,
        device=device,
        dtype=dtype,
    )

    next_request_id = 0
    active_request_ids: list[int] = []
    generated_by_request: dict[int, int] = {}
    request_prompt_lens: dict[int, int] = {}

    prefill_ms = 0.0
    arrival_prefill_ms = 0.0
    arrivals = 0
    finishes = 0

    def add_request(*, initial: bool) -> int:
        nonlocal next_request_id, prefill_ms, arrival_prefill_ms, arrivals
        request_id = next_request_id
        next_request_id += 1
        prompt_len = _sample_prompt_len(
            prompt_lens,
            distribution=config.prompt_length_distribution,
            request_index=request_id,
            generator=generator,
            device=device,
        )
        k, v = _random_kv(
            batch=1,
            heads=config.heads,
            tokens=prompt_len,
            head_dim=config.head_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        synchronize(device)
        start = perf_counter()
        allocator.add_request(request_id, k, v)
        synchronize(device)
        elapsed_ms = (perf_counter() - start) * 1000.0
        if initial:
            prefill_ms += elapsed_ms
        else:
            arrival_prefill_ms += elapsed_ms
        arrivals += 1
        generated_by_request[request_id] = 0
        request_prompt_lens[request_id] = prompt_len
        return request_id

    for _ in range(config.requests):
        active_request_ids.append(add_request(initial=True))

    synchronize(device)
    first_start = perf_counter()
    first_cache = allocator.to_cache(active_request_ids)
    first_q = torch.randn(
        (len(active_request_ids), config.heads, config.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    _ = paged_quant_attention(first_q, first_cache)
    synchronize(device)
    first_attention_ms = (perf_counter() - first_start) * 1000.0
    ttft_ms = prefill_ms + first_attention_ms

    decode_ms = 0.0
    generated_tokens = 0

    for _step in range(config.decode_steps):
        if not active_request_ids:
            break

        synchronize(device)
        decode_start = perf_counter()
        cache = allocator.to_cache(active_request_ids)
        q = torch.randn(
            (len(active_request_ids), config.heads, config.head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        out = paged_quant_attention(q, cache)
        next_k = out.unsqueeze(2).to(dtype=dtype)
        next_v = torch.tanh(out).unsqueeze(2).to(dtype=dtype)
        allocator.append_batch(active_request_ids, next_k, next_v)
        synchronize(device)
        decode_ms += (perf_counter() - decode_start) * 1000.0

        for request_id in active_request_ids:
            generated_by_request[request_id] += 1
        generated_tokens += len(active_request_ids)

        if config.request_life_steps > 0:
            finished_ids = [
                request_id
                for request_id in active_request_ids
                if generated_by_request[request_id] >= config.request_life_steps
            ]
            for request_id in finished_ids:
                allocator.release_request(request_id)
                active_request_ids.remove(request_id)
                generated_by_request.pop(request_id, None)
                finishes += 1
            for _ in finished_ids:
                active_request_ids.append(add_request(initial=False))

    allocator_stats = allocator.stats()
    avg_prompt_len = sum(request_prompt_lens.values()) / max(1, len(request_prompt_lens))

    return {
        "device": str(device),
        "device_name": device_name(device),
        "dtype": str(dtype).replace("torch.", ""),
        "requests": config.requests,
        "prompt_len": config.prompt_len,
        "prompt_lens": list(prompt_lens),
        "prompt_length_distribution": config.prompt_length_distribution,
        "average_prompt_len": avg_prompt_len,
        "decode_steps": config.decode_steps,
        "request_life_steps": config.request_life_steps,
        "arrivals": arrivals,
        "finishes": finishes,
        "prefill_ms": prefill_ms,
        "arrival_prefill_ms": arrival_prefill_ms,
        "total_prefill_ms": prefill_ms + arrival_prefill_ms,
        "first_attention_ms": first_attention_ms,
        "decode_ms": decode_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": decode_ms / max(1, config.decode_steps),
        "tokens_per_second": generated_tokens / max(1.0e-9, decode_ms / 1000.0),
        "generated_tokens": generated_tokens,
        **allocator_stats,
    }


def _validate_config(config: ServingConfig) -> None:
    if config.requests <= 0:
        raise ValueError("requests 必须为正数")
    if config.prompt_len <= 0:
        raise ValueError("prompt_len 必须为正数")
    if config.decode_steps < 0:
        raise ValueError("decode_steps 不能为负数")
    if config.request_life_steps < 0:
        raise ValueError("request_life_steps 不能为负数")
    if config.heads <= 0 or config.head_dim <= 0:
        raise ValueError("heads 和 head_dim 必须为正数")
    if config.block_size <= 0:
        raise ValueError("block_size 必须为正数")


def _resolve_prompt_lens(config: ServingConfig) -> tuple[int, ...]:
    lens = config.prompt_lens if config.prompt_lens is not None else (config.prompt_len,)
    if not lens:
        raise ValueError("prompt_lens 不能为空")
    if any(length <= 0 for length in lens):
        raise ValueError("prompt_lens 必须全部为正数")
    return tuple(int(length) for length in lens)


def _estimate_allocator_blocks(config: ServingConfig, prompt_lens: Sequence[int]) -> int:
    if config.allocator_blocks is not None:
        if config.allocator_blocks <= 0:
            raise ValueError("allocator_blocks 必须为正数")
        return int(config.allocator_blocks)

    max_prompt_len = max(prompt_lens)
    tokens_per_live_request = max_prompt_len + max(1, config.decode_steps)
    blocks_per_live_request = (tokens_per_live_request + config.block_size - 1) // config.block_size
    return max(1, config.requests * blocks_per_live_request)


def _sample_prompt_len(
    prompt_lens: Sequence[int],
    *,
    distribution: str,
    request_index: int,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    mode = distribution.strip().lower()
    if mode in {"uniform", "cycle"}:
        return int(prompt_lens[request_index % len(prompt_lens)])
    if mode == "random":
        idx = int(torch.randint(len(prompt_lens), (1,), generator=generator, device=device).item())
        return int(prompt_lens[idx])
    if mode == "bimodal":
        if len(prompt_lens) == 1:
            return int(prompt_lens[0])
        return int(prompt_lens[0] if request_index % 2 == 0 else prompt_lens[-1])
    raise ValueError("prompt_length_distribution 必须是 uniform、random 或 bimodal")


def _random_kv(
    *,
    batch: int,
    heads: int,
    tokens: int,
    head_dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch, heads, tokens, head_dim)
    k = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return k, v
