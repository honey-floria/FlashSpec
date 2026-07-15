# FlashSpec Docs

这个目录存放 FlashSpec 的技术说明、路线图和后续工程任务。根目录 `README.md` 负责快速开始和命令说明；本目录更偏向设计背景、实现边界和下一步开发计划。

## 文档索引

- `deep_engineering_project.svg`：项目技术地图，概览 decode attention、paged KV cache、quantization、profiling 和 serving loop 之间的关系。
- `TODO.MD`：后续工程任务清单，按优先级拆分为 kernel、paged cache、profiling、serving 和测试几条线。

## 项目目标

FlashSpec 面向 LLM decode 阶段的 KV cache 带宽瓶颈。目标路径是：

1. 将 K/V cache 从 FP16/BF16 压缩为 paged INT8 block。
2. 通过 `block_table` 支持 batch 内不同 request 的分页寻址。
3. 在 attention kernel 内直接读取 INT8 KV，完成反量化、QK、softmax 和 PV。
4. 避免在主路径 materialize dense FP16/FP32 KV。
5. 用 benchmark 和 profiler 证明 tokens/s、TPOT、HBM bandwidth 等指标改善。

## 当前实现状态

当前代码已经实现可移植 PyTorch 后端，用于验证 correctness 和工程流程：

- per-block affine INT8 quant/dequant；
- paged KV cache 数据结构和 `block_table`；
- dense reference attention；
- fused/paged attention API；
- Kernel 1 Triton fused dequant attention，可在 CUDA + Triton 环境中直接读取 INT8 K/V，并在 kernel 内完成反量化、QK、softmax 和 PV；
- Kernel 2 Triton paged INT8 KV attention，可在 CUDA + Triton 环境中读取 `block_table`，直接按 logical block 间接寻址 physical KV block；
- decode serving loop 模拟；
- microbenchmark、sweep、profiling report 和 roofline SVG 脚本。

需要注意：`fused` 是 portable PyTorch 路径，仍会反量化并还原 dense KV，再调用 reference attention；`triton_fused` 是 Kernel 1 的 Triton 路径，主路径不 materialize dense KV。`paged` 是 portable PyTorch 路径，仍会通过 `cache.to_dense()` 还原 KV；`triton_paged` 是 Kernel 2 的 Triton 路径，主路径不 materialize dense KV。

## 关键边界

- benchmark 中的 `estimated_*` bandwidth 字段是根据 KV 字节数和 CUDA event latency 推导的估算值，不是 Nsight Compute 实测值。
- microbenchmark 的 CUDA 路径已经使用 `torch.cuda.Event` 计时，并输出 warmup/repeat 后的 `latency_p50_ms`、`latency_p90_ms`、`latency_p99_ms` 和 raw latency。
- JSON 中的 `measured_kernel_latency_ms` 来自 CUDA event；`measured_dram_bytes`、`measured_achieved_bandwidth_gbps`、`measured_occupancy_pct`、`measured_sm_utilization_pct`、`measured_registers_per_thread`、`measured_theoretical_occupancy_pct` 仍需要运行 JSON 里的 `nsight_compute_command` 后回填。
- JSON 顶层会保留实验配置字段：`num_splits`、`block_n`、`env_flashspec_num_splits`、`env_flashspec_block_n`。优化实验必须优先以这些字段和结果文件为准，避免只凭文件名或 notebook cell 推断运行配置。
- `latency_breakdown` 是 decode attention 的阶段定义和粗粒度工作量估算，覆盖 KV load、dequant、QK、softmax、PV accumulation、output write；真正的每阶段 latency 需要 Nsight Compute source-line/instruction/memory metrics 进一步归因。
- `materializes_dense_kv=true` 表示该路径仍会创建 dense KV 中间结果。
- Triton 相关代码已按功能拆分：`triton_fused.py` 实现 Kernel 1 fused dequant attention，`triton_paged.py` 实现 Kernel 2 paged INT8 KV attention，`triton_utils.py` 负责可选 Triton 导入和共享工具，`triton_kernels.py` 只保留兼容导出入口。
- serving 模拟已经支持 cache append，但 portable append 仍会重新量化物理 block store；真实 serving 还需要 allocator/free list 和 request 生命周期管理。

## Profiling 使用方式

固定一个 shape 跑 microbenchmark，并保存 JSON：

```bash
python benchmarks/microbench.py --backend triton_paged --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --block-size 16 --iters 50 --warmup 10 --repeats 20 --device cuda --dtype float16 --json --include-raw --output results/a100_triton_paged.json --raw-output results/a100_triton_paged_raw.json
```

生成 Markdown profiling report：

```bash
python scripts/profile_report.py results/a100_triton_paged.json --output results/a100_triton_paged_profile.md
```

报告中的 `nsight_compute_command` 用来采集 DRAM bytes、achieved bandwidth、occupancy、SM utilization、registers/thread 和 theoretical occupancy。采集到这些值后，再回填 `measured_*` 字段，才能把“估算 bandwidth”和“硬件实测 profiler 指标”区分开。

## 推荐开发顺序

1. [x] Kernel 1：非 paged INT8 KV fused dequant attention（`triton_fused.py`）。
2. [x] Kernel 2：加入 `block_table` 的 paged INT8 KV attention（`triton_paged.py`）。
3. 扩展 serving 模拟，支持 variable length、block allocator、request 完成和新增。
4. 补齐 CUDA、FP16/BF16、边界 shape 和 benchmark schema 测试。
5. 在真实 A100 环境运行 Nsight Compute，回填 `measured_*` profiler 指标并沉淀实验报告。

Kernel 1、2 的代码路径已经完成并在 CUDA + Triton 下有对齐测试；当前重心在第 3~5 步：把 serving 做得更接近真实系统、扩大测试矩阵，并在真实 A100 上采集 profiler 数据。详细任务见 `TODO.MD`。
