"""解析 Nsight Compute (ncu) 的 CSV 输出，聚合成 FlashSpec 需要的实测指标。

ncu 用 `--csv` 输出的是 long format：每行是 (kernel 实例, metric) 组合，
关键列为 ``ID`` / ``Kernel Name`` / ``Metric Name`` / ``Metric Value`` /
``Metric Unit``。一次 profiling 通常覆盖多个 kernel 实例（warmup 之外的多个
launch，或一个 backend 内部的多个子 kernel），因此这里对所有实例做聚合：

- ``.sum`` 类字节指标：跨实例累加。
- 百分比类指标（occupancy / SM throughput）：按 kernel duration 加权平均，
  比简单平均更能反映真实占比。

有效带宽用 ``总字节 / 总时长`` 计算。注意 GB/s 恰好等于 bytes/ns，所以只要
把字节归一到 byte、时长归一到 ns，直接相除即可得到 GB/s。

本模块不依赖 CUDA / torch，可在任意机器上用合成 CSV 做单元测试。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

# ncu 不同版本 / locale 下的单位写法，统一归一到 byte 与 ns。
_BYTE_UNITS = {
    "byte": 1.0, "bytes": 1.0, "b": 1.0,
    "kbyte": 1e3, "kbytes": 1e3, "kib": 1024.0,
    "mbyte": 1e6, "mbytes": 1e6, "mib": 1024.0**2,
    "gbyte": 1e9, "gbytes": 1e9, "gib": 1024.0**3,
}
_TIME_UNITS_TO_NS = {
    "ns": 1.0, "nsecond": 1.0, "nseconds": 1.0,
    "us": 1e3, "usecond": 1e3, "useconds": 1e3, "µs": 1e3,
    "ms": 1e6, "msecond": 1e6, "mseconds": 1e6,
    "s": 1e9, "second": 1e9, "seconds": 1e9,
}

# 我们关心的 ncu metric -> 内部字段名。
_DRAM_READ = "dram__bytes_read.sum"
_DRAM_WRITE = "dram__bytes_write.sum"
_DURATION = "gpu__time_duration.sum"
_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"
_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_DRAM_THROUGHPUT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"


def _to_float(raw: str) -> float | None:
    """把 ncu 的数值字符串转 float，容忍千分位逗号与空串。"""
    if raw is None:
        return None
    text = raw.strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize(value: float, unit: str, table: dict[str, float]) -> float:
    """按单位表把 value 归一；未知单位按 1.0 处理（并保持原值）。"""
    return value * table.get(unit.strip().lower(), 1.0)


@dataclass
class _Kernel:
    """单个 kernel 实例聚合到的原始指标。"""

    name: str = ""
    dram_bytes: float = 0.0
    duration_ns: float = 0.0
    occupancy_pct: float | None = None
    sm_pct: float | None = None
    dram_pct: float | None = None


@dataclass
class NcuMetrics:
    """跨全部 kernel 实例聚合后的实测指标，供回填 JSON 使用。"""

    dram_bytes: float
    duration_ns: float
    achieved_bandwidth_gbps: float
    occupancy_pct: float | None
    sm_utilization_pct: float | None
    dram_throughput_pct: float | None
    kernel_count: int
    kernel_names: list[str] = field(default_factory=list)

    def as_backfill(self) -> dict[str, object]:
        """转成 microbench JSON 里的 measured_* 字段字典。"""
        return {
            "measured_dram_bytes": self.dram_bytes,
            "measured_achieved_bandwidth_gbps": self.achieved_bandwidth_gbps,
            "measured_occupancy_pct": self.occupancy_pct,
            "measured_sm_utilization_pct": self.sm_utilization_pct,
            "measured_dram_throughput_pct": self.dram_throughput_pct,
            "measured_ncu_kernel_duration_ms": self.duration_ns / 1e6,
            "measured_ncu_kernel_count": self.kernel_count,
            "measured_ncu_kernel_names": self.kernel_names,
            "profiler_metrics_source": "nsight_compute_csv",
        }


def _find_col(header: list[str], *candidates: str) -> int | None:
    """在 header 里按候选名（大小写不敏感）找列索引。"""
    lowered = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered.index(cand.lower())
    return None


def parse_ncu_csv(text: str) -> NcuMetrics:
    """解析 ncu --csv 输出文本，聚合成 NcuMetrics。

    ncu 在 metric 行之前可能打印告警/横幅，所以先定位真正的表头行
    （包含 "Metric Name" 与 "Metric Value" 的那一行）。
    """

    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    header_idx = None
    for i, row in enumerate(rows):
        low = [c.strip().lower() for c in row]
        if "metric name" in low and "metric value" in low:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("在 ncu CSV 中找不到表头（缺少 'Metric Name'/'Metric Value' 列）")

    header = rows[header_idx]
    id_col = _find_col(header, "ID")
    name_col = _find_col(header, "Kernel Name")
    metric_col = _find_col(header, "Metric Name")
    value_col = _find_col(header, "Metric Value")
    unit_col = _find_col(header, "Metric Unit")
    if metric_col is None or value_col is None:
        raise ValueError("ncu CSV 缺少 Metric Name / Metric Value 列")

    kernels: dict[str, _Kernel] = {}
    order: list[str] = []
    for row in rows[header_idx + 1:]:
        if len(row) <= max(metric_col, value_col):
            continue
        metric = row[metric_col].strip()
        value = _to_float(row[value_col])
        if value is None:
            continue
        unit = row[unit_col] if unit_col is not None and unit_col < len(row) else ""
        key = row[id_col].strip() if id_col is not None and id_col < len(row) else "0"
        if key not in kernels:
            kernels[key] = _Kernel()
            order.append(key)
        k = kernels[key]
        if name_col is not None and name_col < len(row) and not k.name:
            k.name = row[name_col].strip()
        if metric in (_DRAM_READ, _DRAM_WRITE):
            k.dram_bytes += _normalize(value, unit, _BYTE_UNITS)
        elif metric == _DURATION:
            k.duration_ns += _normalize(value, unit, _TIME_UNITS_TO_NS)
        elif metric == _OCCUPANCY:
            k.occupancy_pct = value
        elif metric == _SM_THROUGHPUT:
            k.sm_pct = value
        elif metric == _DRAM_THROUGHPUT:
            k.dram_pct = value

    if not kernels:
        raise ValueError("ncu CSV 未解析到任何 kernel 指标行")
    return _aggregate([kernels[k] for k in order])


def _weighted(pairs: list[tuple[float | None, float]]) -> float | None:
    """按权重（kernel duration）聚合百分比；权重全 0 时退化为简单平均。"""
    vals = [(v, w) for v, w in pairs if v is not None]
    if not vals:
        return None
    total_w = sum(w for _, w in vals)
    if total_w <= 0:
        return sum(v for v, _ in vals) / len(vals)
    return sum(v * w for v, w in vals) / total_w


def _aggregate(kernels: list[_Kernel]) -> NcuMetrics:
    """把多个 kernel 实例聚合成整体实测指标。"""
    total_bytes = sum(k.dram_bytes for k in kernels)
    total_ns = sum(k.duration_ns for k in kernels)
    # GB/s == bytes/ns，故直接相除。
    bw = total_bytes / total_ns if total_ns > 0 else 0.0
    return NcuMetrics(
        dram_bytes=total_bytes,
        duration_ns=total_ns,
        achieved_bandwidth_gbps=bw,
        occupancy_pct=_weighted([(k.occupancy_pct, k.duration_ns) for k in kernels]),
        sm_utilization_pct=_weighted([(k.sm_pct, k.duration_ns) for k in kernels]),
        dram_throughput_pct=_weighted([(k.dram_pct, k.duration_ns) for k in kernels]),
        kernel_count=len(kernels),
        kernel_names=[k.name for k in kernels if k.name],
    )
