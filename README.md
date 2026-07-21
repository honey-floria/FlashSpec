# FlashSpec

FlashSpec 是一个面向 LLM decode 阶段的 kernel/profiling 工程项目。目标是把 KV cache 存成 paged INT8 blocks，并在 attention kernel 内直接完成反量化、QK、softmax 和 PV，减少 decode attention 对 HBM 带宽的压力。

当前项目已经完成两条 Triton 主路径：

- Kernel 1: `triton_fused`，连续 INT8 KV 的 fused dequant attention。
- Kernel 2: `triton_paged`，通过 `block_table` 间接寻址的 paged INT8 KV attention。

另外还有一个 allocator-backed serving 模拟层，用来验证 request lifecycle、prefill/decode 分离、block 复用和 fragmentation。

这不是生产级 serving runtime。它现在更准确地说是一个可复现的 kernel optimization case study：包含 correctness、microbenchmark、A100 profiling matrix、Nsight Compute fast metrics 和优化日志。

## 当前结论

截至 2026-07-15，Colab A100 full matrix + NCU 已跑完：

- `triton_fused`: 48 个点，覆盖 `block_n={32,64,128}`、`num_warps={4,8}`、`num_splits={auto,1,4,8}`。
- `triton_paged`: 72 个点，覆盖 `block_n={32,64,128}`、`num_warps={4,8}`、`length_pattern={uniform,descending}`、`paged_layout={contiguous,shuffled,interleaved}`。
- 120 个点全部有 `measured_achieved_bandwidth_gbps`、DRAM throughput、occupancy 和 registers/thread，无 `profiler_error`。

当前最稳的默认候选是：

```text
block_n = 128
num_warps = 4
num_splits = 4  # triton_fused
```

关键结果：

| backend | 场景 | 最优配置 | latency |
|---|---|---|---:|
| `triton_fused` | s2048/d128 | `block_n=128, num_warps=4, split=4` | 0.2555 ms |
| `triton_fused` | s4096/d128 | `block_n=128, num_warps=4, split=4` | 0.4856 ms |
| `triton_paged` | s2048/d128 uniform | `block_n=128, num_warps=4, contiguous` | 0.2859 ms |
| `triton_paged` | s4096/d128 uniform | `block_n=128, num_warps=4, contiguous` | 0.5604 ms |

结论：

- `block_n=32` 不适合作为默认。它降低寄存器、提高 occupancy，但 latency 和 DRAM throughput 明显变差。
- `num_warps=8` 也不适合作为默认。它通常比 `num_warps=4` 慢。
- paged uniform 相对 fused 最优慢约 12-15%，这是当前 block_table/paged path 的主要优化空间。
- source-line / instruction attribution 已确认：继续追 occupancy 不是主线，下一步应减少 MIO/scoreboard 等待和 paged 地址计算开销。

完整实验过程和历史结论见 [doc/optimization-log.md](doc/optimization-log.md)。

## 结果速查表

以下速查表汇总 Colab A100 full matrix（120 点）的关键参数结果，方便快速对比。完整逐点数据见 [results/profile_matrix_report.md](results/profile_matrix_report.md)。

### 各后端最优点（s2048 / s4096, d128）

| backend | seq | 最优配置 | latency ms | BW GB/s | DRAM % | occ % | regs |
|---|---:|---|---:|---:|---:|---:|---:|
| `triton_fused` | 2048 | `block_n=128, warps=4, split=4, uniform, contiguous` | 0.2555 | 918.9 | 59.1 | 18.0 | 168 |
| `triton_fused` | 4096 | `block_n=128, warps=4, split=4, uniform, contiguous` | 0.4856 | 931.6 | 59.9 | 18.0 | 168 |
| `triton_paged` | 2048 | `block_n=128, warps=4, uniform, contiguous` | 0.2858 | 790.7 | 50.8 | 15.5 | 168 |
| `triton_paged` | 4096 | `block_n=128, warps=4, uniform, contiguous` | 0.5604 | 789.4 | 50.8 | 15.5 | 168 |

### block_n × num_warps 平均（triton_fused）

| block_n | num_warps | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 8 | 0.3880 | 900.7 | 57.9 | 17.5 | 168 |
| 128 | 8 | 8 | 0.5671 | 605.8 | 39.0 | 24.1 | 119 |
| 64 | 4 | 8 | 0.4478 | 771.8 | 49.6 | 24.1 | 110 |
| 64 | 8 | 8 | 0.7486 | 448.2 | 28.8 | 26.3 | 106 |
| 32 | 4 | 8 | 0.5910 | 567.6 | 36.5 | 36.7 | 74 |
| 32 | 8 | 8 | 1.1275 | 294.7 | 19.0 | 35.2 | 80 |

### num_splits 平均（triton_fused）

| num_splits | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.6690 | 566.3 | 36.4 | 25.3 | 104 |
| 4 | 18 | 0.5733 | 605.2 | 38.9 | 27.6 | 111 |
| 8 | 18 | 0.7007 | 612.3 | 39.4 | 28.3 | 111 |

### block_n × num_warps 平均（triton_paged）

| block_n | num_warps | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 12 | 0.4284 | 657.2 | 42.3 | 15.5 | 168 |
| 128 | 8 | 12 | 0.5588 | 491.0 | 31.6 | 22.7 | 128 |
| 64 | 4 | 12 | 0.4876 | 574.9 | 37.0 | 26.0 | 96 |
| 64 | 8 | 12 | 0.8391 | 322.0 | 20.7 | 22.8 | 116 |
| 32 | 4 | 12 | 0.6789 | 396.5 | 25.5 | 27.4 | 72 |
| 32 | 8 | 12 | 1.2063 | 223.1 | 14.3 | 31.6 | 80 |

### paged_layout 平均（triton_paged）

| paged_layout | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---|---:|---:|---:|---:|---:|---:|
| contiguous | 24 | 0.6935 | 443.7 | 28.5 | 24.3 | 110 |
| interleaved | 24 | 0.6966 | 446.0 | 28.7 | 24.3 | 110 |
| shuffled | 24 | 0.7095 | 442.7 | 28.5 | 24.3 | 110 |

速查表要点：`block_n=128 + num_warps=4` 在两条路径上都是最快组合；`num_splits=4` 在 fused 上最优；paged 三种 layout 平均 latency 几乎持平（contiguous 略优），说明当前 paged 开销主要来自 block_table 地址计算而非物理布局局部性。

## 安装

基础安装：

```bash
python -m pip install -e .
```

Triton 路径：

```bash
python -m pip install -e ".[triton]"
```

## 正确性测试

```bash
python -m unittest discover -s tests
```

测试覆盖：

- INT8 quant/dequant round trip；
- dense reference attention；
- portable fused/paged reference；
- paged KV cache 和 `block_table`；
- append 后的 paged cache correctness；
- CUDA + Triton 下的 `triton_fused` / `triton_paged` correctness，无 GPU 时自动 skip；
- microbench JSON schema 的关键字段。

## 后端

| backend | 说明 | 是否分页 | 是否 materialize dense KV |
|---|---|---:|---:|
| `dense` | FP16/FP32 dense reference attention | 否 | 否 |
| `fused` | portable PyTorch INT8 reference，先反量化再 dense attention | 否 | 是 |
| `triton_fused` | Triton Kernel 1，kernel 内读 INT8 KV 并反量化 | 否 | 否 |
| `paged` | portable PyTorch paged reference，通过 `cache.to_dense()` 验证 | 是 | 是 |
| `triton_paged` | Triton Kernel 2，kernel 内通过 `block_table` 读 physical INT8 blocks | 是 | 否 |

`fused` 和 `paged` 是 correctness/reference 路径；真实性能结论应优先看 `triton_fused` 和 `triton_paged`。

## Colab A100 流程

推荐直接使用 [run.ipynb](run.ipynb)。顺序是：

1. 挂载 Google Drive。
2. 检查 A100 / CUDA / Triton / Nsight Compute。
3. 跑 correctness tests。
4. 跑 sanity profiling。
5. 跑 matrix profiling。
6. 汇总 manifest 和图表。
7. 对关键 JSON 生成 profile report 或 source-line 命令。

矩阵 profiling 的 notebook 开关：

```python
MATRIX_PRESET = "dry"    # 只打印命令
MATRIX_PRESET = "small"  # 关键组合
MATRIX_PRESET = "full"   # 完整矩阵

PROFILE_NCU = True       # 采集 Nsight Compute fast metrics，慢但有硬件指标
PROFILE_NCU = False      # 只看 latency，适合快速筛候选
```

结果位置：

```text
results/profile_matrix/fused/triton_fused_manifest.csv
results/profile_matrix/paged/triton_paged_manifest.csv
results/profile_matrix_report.md
```

普通分析和默认参数判断只看这两个 manifest、`results/profile_matrix_report.md` 和 `results/ncu_source_attribution_export/`。`results/colab_kernels/*.json` 是 2026-07-15 的历史单点 microbench 样本，用来复现旧图表或生成单点 `profile_report`，不要当作当前默认配置的验收依据。

## 常用命令

单点 microbenchmark：

```bash
python benchmarks/microbench.py \
  --backend triton_fused \
  --batch 16 --heads 32 --seq-len 2048 --head-dim 128 \
  --block-size 16 --iters 50 --warmup 10 --repeats 20 \
  --device cuda --dtype float16 \
  --json --include-raw --profile-ncu \
  --output results/triton_fused_s2048_d128.json
```

full matrix：

```bash
python scripts/profile_matrix.py --backend triton_fused \
  --seq-lens 2048,4096 --head-dims 128 \
  --block-ns 32,64,128 --num-warps 4,8 \
  --num-splits auto,1,4,8 \
  --length-patterns uniform \
  --profile-ncu --output-dir results/profile_matrix/fused

python scripts/profile_matrix.py --backend triton_paged \
  --seq-lens 2048,4096 --head-dims 128 \
  --block-ns 32,64,128 --num-warps 4,8 \
  --length-patterns uniform,descending \
  --paged-layouts contiguous,shuffled,interleaved \
  --profile-ncu --output-dir results/profile_matrix/paged
```

生成稳定矩阵报告：

```bash
python scripts/analyze_matrix.py \
  --matrix-dir results/profile_matrix \
  --source-dir results/ncu_source_attribution_export \
  --output results/profile_matrix_report.md
```

汇总历史单点 JSON 图表：

```bash
python scripts/analyze_results.py \
  --results-dir results/colab_kernels \
  --output-dir results/colab_kernels/analysis
```

这会让 `results/colab_kernels/analysis/summary.csv` 与同目录 JSON 保持一致；它不替代 full matrix 报告。

生成单点 profile report：

```bash
python scripts/profile_report.py \
  results/colab_kernels/triton_fused_s2048_d128.json \
  --output results/colab_kernels/triton_fused_s2048_d128_profile.md
```

batch/seq sweep：

```bash
python benchmarks/sweep.py \
  --backends dense,triton_fused,triton_paged \
  --batches 1,4,8,16 \
  --seq-lens 512,1024,2048,4096 \
  --heads 32 --head-dim 128 \
  --iters 20 --warmup 5 --repeats 10 \
  --output results/a100_sweep.csv
```

serving allocator benchmark：

```bash
python benchmarks/e2e_serving.py \
  --requests 32 --prompt-lens 512,1024,2048 \
  --prompt-length-distribution bimodal \
  --decode-steps 64 --request-life-steps 32 \
  --heads 32 --head-dim 128 --json
```

JSON 输出会区分初始 prefill、后续 arrival prefill 和 decode loop，并报告
allocator 指标，例如 `ttft_ms`、`prefill_ms`、`arrival_prefill_ms`、
`decode_ms`、`tpot_ms`、`tokens_per_second`、`block_utilization` 和
`fragmentation`。

## 项目结构

```text
src/flashspec/
  attention.py       dense / 量化 decode attention API
  quant.py           按 block 的仿射 INT8 量化
  paged.py           paged quant-KV cache、allocator 和 block_table
  runtime.py         设备 / dtype 辅助函数
  serving.py         allocator 驱动的 decode serving loop
  ncu_parse.py       Nsight Compute CSV 解析器
  triton_fused.py    Kernel 1：fused INT8 KV attention
  triton_paged.py    Kernel 2：paged INT8 KV attention
  triton_utils.py    可选 Triton 导入辅助
  triton_kernels.py  兼容性导出

benchmarks/
  microbench.py      kernel 延迟 / 带宽 / NCU 采集
  sweep.py           batch x seq_len 扫描
  e2e_serving.py     serving allocator benchmark

scripts/
  profile_matrix.py            矩阵 profiling 执行器
  analyze_results.py           单点 JSON 汇总和图表
  profile_report.py            单个 JSON 的 Markdown 报告
  backfill_ncu.py              将实测 Nsight Compute 字段回填到 JSON
  export_ncu_source_reports.py 导出 .ncu-rep 页面用于 source attribution
  analyze_matrix.py            矩阵和 source attribution 的稳定 Markdown 报告

doc/
  optimization-log.md 实验历史和 profiling 结论
  project-flow.md     项目流程图和工程闭环说明
  TODO.MD             剩余工作和里程碑
```

## 当前边界

- full matrix 目前主要覆盖 `head_dim=128`、`seq_len={2048,4096}`，还需要补 `head_dim=64`、短序列和更多 request length 分布。
- serving benchmark 使用 allocator 模拟 free list、request lifecycle 和 fragmentation，但仍使用随机 tensor，不代表真实模型 KV 分布。
- `latency_breakdown` 是阶段定义和工作量估算，不是每阶段真实耗时；关键 s2048/s4096 点已补 source-line / instruction / memory workload 归因。
- 随机 tensor 不能代表真实模型 KV 分布；量化误差还需要真实模型 KV sample workflow。
- Kernel 3（tree-attn / speculative decode）仍是可选 bonus，后续按 `doc/TODO.MD` 的步骤推进。

## 下一步

1. 继续做 attribution-driven kernel patch：减少 shared staging / dequant / softmax 更新里的 MIO、short scoreboard，以及 paged block_table 地址计算。
2. 补真实模型 KV sample workflow，验证量化误差和随机 tensor profiling 的差异。
3. 补测试矩阵和 CI。
4. 按 `doc/TODO.MD` 里的步骤推进 Kernel 3 tree-attn / speculative decode。
