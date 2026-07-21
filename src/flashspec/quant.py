from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class QuantizedTensor:
    """按连续 sequence block 做仿射 INT8 量化后的张量。

    量化方式是 affine/asymmetric quantization：
    - 先在每个 block 内统计最小值 block_min 和最大值 block_max。
    - 用 scale = (block_max - block_min) / 255 把浮点范围映射到 uint8 [0, 255]。
    - 用 zero_point 表示浮点 0 在 uint8 域中的位置。
    - 为了用 torch.int8 保存，实际 values 存的是 uint8 值减去 128 后的结果。

    字段说明：
    - values: 量化后的 int8 数据。逻辑上对应 uint8 值域 [0, 255]，
      但实际存储为 signed int8，因此反量化时需要先 `values + 128`。
    - scale: 每个 block 的缩放系数，用于把整数值还原到浮点范围。
    - zero_point: 每个 block 的零点，存储在 uint8 域，实际 dtype 使用 int16。
    - block_size: sequence 维度上每个量化 block 包含的 token 数。
    - original_shape: 量化前的原始形状，用于反量化后去掉 padding 并恢复布局。

    约定：
    - sequence 维是倒数第 2 维。
    - feature/head_dim 维是最后 1 维。
    - 所有 leading 维度都会被保留，并分别独立计算 block 量化参数。
    """

    values: torch.Tensor  # 量化后的 int8 数据，布局通常是 [..., n_blocks, block_size, head_dim]。
    scale: torch.Tensor  # 每个 block 的 scale，形状可广播到 values。
    zero_point: torch.Tensor  # 每个 block 的 zero point，位于 uint8 域，但用 int16 存储便于计算。
    block_size: int  # sequence 维度上每个 block 的 token 数。
    original_shape: Tuple[int, ...]  # 量化前张量的真实形状；反量化时依赖它裁剪 padding。

    @property
    def device(self) -> torch.device:
        """量化数据所在设备。"""

        return self.values.device


def _validate_block_tensor(x: torch.Tensor, block_size: int) -> None:
    """校验可按 sequence block 量化的输入张量。

    参数：
    - x: 待量化张量，至少需要 sequence 维和 feature 维两维。
    - block_size: sequence 维度上的 block 大小，必须为正数。

    这里不限制 x 的 dtype；真正量化前会统一转换成 float32。
    """

    # 至少需要倒数第 2 维作为 sequence，最后 1 维作为 feature/head_dim。
    if x.ndim < 2:
        raise ValueError("输入张量至少需要 sequence 和 feature 两个维度")

    # block_size 不能为 0 或负数，否则无法分块。
    if block_size <= 0:
        raise ValueError("block_size 必须为正数")

    # sequence 维和 feature 维都不能为空。
    if x.shape[-1] <= 0 or x.shape[-2] <= 0:
        raise ValueError("sequence 和 feature 维度不能为空")


def _pad_sequence_dim(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """在 sequence 维度补零，使长度可以被 block_size 整除。

    参数：
    - x: 待 padding 张量，sequence 维是倒数第 2 维。
    - block_size: 分块大小。

    返回：
    - 如果 seq_len 已经能被 block_size 整除，直接返回原张量。
    - 否则在 sequence 维末尾补 0，便于 reshape 成完整 block。
    """

    # 原始 sequence 长度。
    seq_len = x.shape[-2]

    # 需要补齐的 token 数。若已经整除，则 pad 为 0。
    pad = (block_size - seq_len % block_size) % block_size
    if pad == 0:
        return x

    # padding 部分的形状保持所有 leading 维和 feature 维不变，只增加 sequence 维。
    pad_shape = (*x.shape[:-2], pad, x.shape[-1])
    return torch.cat([x, x.new_zeros(pad_shape)], dim=-2)


def quantize_int8_per_block(x: torch.Tensor, block_size: int = 16) -> QuantizedTensor:
    """按 sequence block 对张量做 affine INT8 量化。

    参数：
    - x: 待量化张量。sequence 维是 -2，feature/head_dim 维是 -1。
    - block_size: 每个量化 block 在 sequence 维度上覆盖的 token 数。

    返回：
    - QuantizedTensor，包含 int8 values、每个 block 的 scale/zero_point、
      block_size 以及原始形状。

    量化参数的统计范围：
    - 对每组 leading index，按 [block_size, feature_dim] 这个二维区域
      计算一个 scale 和 zero_point。
    - 例如输入形状 [batch, heads, seq_len, head_dim] 时，每个
      [batch, head, logical_block] 都有独立的量化参数。
    """

    # 检查 x 至少有 sequence 和 feature 维，且 block_size 合法。
    _validate_block_tensor(x, block_size)

    # 记录量化前的真实形状；反量化时会用它裁剪 padding 并恢复 seq_len。
    original_shape = tuple(x.shape)

    # detach 避免把量化过程接进 autograd；float32 用于稳定计算 min/max/scale。
    # 如果 seq_len 不能被 block_size 整除，这里会在末尾补零。
    x_float = _pad_sequence_dim(x.detach().to(torch.float32), block_size)

    # leading: 除 sequence 和 feature 维以外的所有前缀维度。
    # 对这些维度上的每个位置都会独立分块量化。
    leading = x_float.shape[:-2]

    # padding 后的 sequence 长度，一定能被 block_size 整除。
    seq_len = x_float.shape[-2]

    # feature/head_dim 维度大小。
    head_dim = x_float.shape[-1]

    # 每条 sequence 被切成多少个 block。
    n_blocks = seq_len // block_size

    # blocked: [...leading, n_blocks, block_size, head_dim]。
    # 后续会在每个 block 的 [block_size, head_dim] 范围内统计 min/max。
    blocked = x_float.reshape(*leading, n_blocks, block_size, head_dim)

    # block_min/block_max: 每个 block 内的最小/最大浮点值。
    # keepdim=True 让它们保留 [..., n_blocks, 1, 1] 形状，便于广播到 blocked。
    block_min = blocked.amin(dim=(-2, -1), keepdim=True)
    block_max = blocked.amax(dim=(-2, -1), keepdim=True)

    # scale 把浮点动态范围映射到 256 个 uint8 桶。
    # clamp_min 防止 block 内所有值相同导致 scale 为 0。
    scale = ((block_max - block_min) / 255.0).clamp_min(1.0e-8)

    # zero_point 是浮点 0 映射到 uint8 域的位置。
    # clamp 到 [0, 255]，保证整数域合法。
    zero_point = torch.round(-block_min / scale).clamp(0, 255)

    # q_uint8: 仿射量化后的 uint8 逻辑值，范围 [0, 255]。
    # 公式：q = round(x / scale + zero_point)。
    q_uint8 = torch.round(blocked / scale + zero_point).clamp(0, 255)

    # PyTorch 的 int8 是 signed [-128, 127]，这里用 q_uint8 - 128 存储。
    # 反量化时会先 +128 回到 uint8 域。
    q_int8 = (q_uint8.to(torch.int16) - 128).to(torch.int8)
    return QuantizedTensor(
        values=q_int8,
        scale=scale,
        zero_point=zero_point.to(torch.int16),
        block_size=block_size,
        original_shape=original_shape,
    )


def dequantize_int8_per_block(q: QuantizedTensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """反量化由 quantize_int8_per_block 生成的 QuantizedTensor。

    参数：
    - q: 量化后的张量对象，包含 values、scale、zero_point 和原始形状。
    - dtype: 可选输出 dtype；如果为 None，则返回 float32。

    返回：
    - 反量化后的 dense 张量，形状恢复为 q.original_shape。

    反量化公式：
    - q_uint8 = values + 128
    - x = (q_uint8 - zero_point) * scale
    """

    # values 以 signed int8 存储，需要先转 int16 防止 +128 溢出。
    q_uint8 = q.values.to(torch.int16) + 128

    # 使用 affine 反量化公式还原到 float32。
    # scale 和 zero_point 的形状可广播到 q_uint8。
    x = (q_uint8.to(torch.float32) - q.zero_point.to(torch.float32)) * q.scale.to(torch.float32)

    # leading 是原始张量除 sequence 和 feature 之外的前缀维度。
    leading = q.original_shape[:-2]

    # 原始 sequence 长度；量化时可能补过 padding，反量化后需要裁剪回这个长度。
    seq_len = q.original_shape[-2]

    # 原始 feature/head_dim 维度。
    head_dim = q.original_shape[-1]

    # 当前 x 布局是 [...leading, n_blocks, block_size, head_dim]。
    # reshape 把 n_blocks 和 block_size 合并回 sequence 维，再裁剪掉尾部 padding。
    x = x.reshape(*leading, -1, head_dim)[..., :seq_len, :]

    # 如果指定 dtype，则把输出转换到目标 dtype；否则保留 float32。
    return x.to(dtype=dtype) if dtype is not None else x


def build_compression_stats(
    dense_bytes: float,
    quant_bytes: float,
    *,
    materializes_dense_kv: bool,
    **extra: float,
) -> dict[str, float]:
    """构造 benchmark/report 用的压缩统计字典（4 条 attention 路径共用）。

    公共字段：dense_kv_bytes / quant_kv_bytes / compression_ratio /
    materializes_dense_kv。compression_ratio 用 max(1, quant_bytes) 兜底除零。
    额外的后端专有指标（physical_blocks、num_splits、block_n、num_warps 等）
    通过 **extra 传入，直接并入返回字典。
    """

    stats = {
        "dense_kv_bytes": float(dense_bytes),
        "quant_kv_bytes": float(quant_bytes),
        "compression_ratio": float(dense_bytes / max(1, quant_bytes)),
        "materializes_dense_kv": 1.0 if materializes_dense_kv else 0.0,
    }
    stats.update({key: float(value) for key, value in extra.items()})
    return stats


def estimate_quantized_bytes(q: QuantizedTensor) -> int:
    """估算 QuantizedTensor 的近似存储字节数。

    统计范围：
    - values: int8 数据，每个元素约 1 字节。
    - scale: float32 数据，每个元素约 4 字节。
    - zero_point: int16 数据，每个元素约 2 字节。

    该值用于 benchmark/report 的压缩率估算，不包含 PyTorch tensor
    allocator 元数据、对齐开销或缓存碎片。
    """

    # q.values.numel(): int8 values 字节数。
    # q.scale.numel() * 4: float32 scale 字节数。
    # q.zero_point.numel() * 2: int16 zero_point 字节数。
    return q.values.numel() + q.scale.numel() * 4 + q.zero_point.numel() * 2
