# FlashSpec

FlashSpec 是一个面向 LLM decode 阶段的 kernel/profiling 工程项目。目标是把 KV cache 存成 paged INT8 blocks，并在 attention kernel 内直接完成反量化、QK、softmax 和 PV，减少 decode attention 对 HBM 带宽的压力。

当前项目已经完成两条 Triton 主路径：

- Kernel 1: `triton_fused`，连续 INT8 KV 的 fused dequant attention。
- Kernel 2: `triton_paged`，通过 `block_table` 间接寻址的 paged INT8 KV attention。

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
- 下一步 profiling 应进入 source-line / instruction attribution，而不是继续盲目扩大参数矩阵。

完整实验过程和历史结论见 [doc/optimization-log.md](doc/optimization-log.md)。

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
```

普通分析只需要这两个 manifest。只有要做单点 `profile_report` 或 source-line 归因时，才需要对应 JSON。

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

汇总单点 JSON 图表：

```bash
python scripts/analyze_results.py \
  --results-dir results/colab_kernels \
  --output-dir results/colab_kernels/analysis
```

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

serving smoke benchmark：

```bash
python benchmarks/e2e_serving.py \
  --requests 32 --prompt-len 1024 --decode-steps 64 \
  --heads 32 --head-dim 128 --json
```

## 项目结构

```text
src/flashspec/
  attention.py       dense / quantized decode attention API
  quant.py           per-block affine INT8 quantization
  paged.py           paged quant-KV cache and block_table
  runtime.py         device / dtype helpers
  serving.py         minimal decode serving loop
  ncu_parse.py       Nsight Compute CSV parser
  triton_fused.py    Kernel 1: fused INT8 KV attention
  triton_paged.py    Kernel 2: paged INT8 KV attention
  triton_utils.py    optional Triton import helpers
  triton_kernels.py  compatibility exports

benchmarks/
  microbench.py      kernel latency / bandwidth / NCU collection
  sweep.py           batch x seq_len sweep
  e2e_serving.py     minimal serving benchmark

scripts/
  profile_matrix.py  matrix profiling runner
  analyze_results.py single-point JSON aggregation and plots
  profile_report.py  Markdown report for one JSON

doc/
  optimization-log.md experiment history and profiling conclusions
  TODO.MD             remaining work and milestones
```

## 当前边界

- full matrix 目前主要覆盖 `head_dim=128`、`seq_len={2048,4096}`，还需要补 `head_dim=64`、短序列和更多 request length 分布。
- `triton_paged` 已有真实 paged KV path，但 serving allocator 仍是简化版，还没有完整 free list、request lifecycle 和 fragmentation 统计。
- `latency_breakdown` 是阶段定义和工作量估算，不是每阶段真实耗时。真实归因需要 source-line / instruction / memory workload metrics。
- 随机 tensor 不能代表真实模型 KV 分布；量化误差还需要真实模型 KV sample workflow。

## 下一步

1. 将默认 kernel 参数收敛到 `block_n=128, num_warps=4, split=4`，并保留 env override。
2. 对 fused 最优点、paged 最优点和 paged shuffled 慢点跑 source-line / instruction 归因。
3. 增加 matrix manifest 自动分析脚本，自动输出 top-k 和参数对比。
4. 补 serving allocator、prefill/decode 分离、block utilization 和 fragmentation。
5. 补测试矩阵和 CI。
