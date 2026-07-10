# FlashSpec

FlashSpec 是一个面向 LLM decode 阶段的工程项目。它的目标是针对 decode attention 的显存带宽瓶颈，把 KV cache 存成 paged INT8 block，并在 attention 路径里融合反量化，减少 FP16 KV 从 HBM 到 SM 的搬运量。

本仓库对应 `doc/deep_engineering_project.svg` 里的技术地图：

- Kernel 1：INT8 KV 的 fused dequant-attention。
- Kernel 2：带 `block_table` 间接寻址的 paged quant-KV attention。
- Profiling：roofline 输入、latency breakdown、microbenchmark。
- Serving：一个最小 decode loop，用来跑通 paged cache 路径。

当前代码包含可移植 PyTorch 后端，方便在 CPU 和 CUDA 上验证 correctness、分页数据结构和 benchmark 流程。PyTorch 后端会 materialize dense KV；Kernel 1 已提供可选 Triton fused dequant attention 后端，在 CUDA + Triton 环境中直接读取 INT8 KV 并在 kernel 内完成反量化、QK、softmax 和 PV。Kernel 2 也已提供可选 Triton paged attention 后端，在 kernel 内读取 `block_table` 并直接寻址 physical INT8 KV block。

## 后端一览

| backend | 说明 | 设备要求 | `materializes_dense_kv` |
|---|---|---|---|
| `dense` | FP16/FP32 dense reference attention，作为 baseline | CPU / CUDA | false |
| `fused` | Kernel 1 的 portable PyTorch 参考实现，先反量化再走 dense attention | CPU / CUDA | true |
| `triton_fused` | Kernel 1 的 Triton 实现，kernel 内直接读 INT8 KV | CUDA + Triton | false |
| `paged` | Kernel 2 的 portable PyTorch 参考实现，先 `to_dense()` 还原 KV | CPU / CUDA | true |
| `triton_paged` | Kernel 2 的 Triton 实现，kernel 内按 `block_table` 间接寻址 | CUDA + Triton | false |

在没有 CUDA/Triton 的环境里，`triton_fused` / `triton_paged` 的 Python 入口会自动回退到对应 portable 实现，保证代码可导入、可 smoke test；benchmark 脚本则会显式报错，避免把 CPU 数字误当成 Triton kernel 性能。

## 当前实现边界

- `fused` 后端是 portable PyTorch 参考实现，会 materialize dense KV；`triton_fused` 后端是 Kernel 1 的 Triton 实现，CUDA 主路径不 materialize dense KV。
- `paged` 后端是 portable PyTorch 参考实现，仍会通过 `cache.to_dense()` 还原 KV；`triton_paged` 后端是 Kernel 2 的 Triton 实现，会在 kernel 内按 `block_table` 间接寻址 physical block。
- microbenchmark 使用 CUDA event 记录 GPU latency，输出 `latency_p50_ms`、`latency_p90_ms`、raw measurements、`estimated_*` 和 `measured_*` 字段。KV bandwidth 仍是基于字节数和 latency 的估算值，JSON 中的 `bandwidth_fields_are_estimates` 会标记这一点；DRAM bytes、occupancy、SM utilization 需要用 JSON 中的 Nsight Compute 命令采集后回填。
- `materializes_dense_kv=true` 表示该后端在当前 PyTorch 实现里会先还原 dense KV，再执行 reference attention。
- serving 模拟使用 paged cache 的增量 `append` 路径，不再在每个 decode step 从完整 dense KV 重新构建 cache；但 portable 后端仍会为了 correctness 反量化物理 block。

## Colab A100 快速开始

在 Colab 里先选择 `Runtime -> Change runtime type -> A100 GPU`，然后运行：

```bash
git clone <your-repo-url> FlashSpec
cd FlashSpec
python -m pip install -e .
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

如果你把项目文件直接上传到了 Colab 当前目录，从 `python -m pip install -e .` 开始执行即可。

## 安装

基础安装：

```bash
python -m pip install -e .
```

如果要继续写真正的 Triton kernel：

```bash
python -m pip install -e ".[triton]"
```

## 正确性测试

```bash
python -m unittest discover -s tests
```

测试覆盖：

- 量化/反量化 round trip 的误差边界；
- fused dequant attention 对齐 dense attention；
- Triton fused/paged 入口在 CPU 上回退并对齐 portable 实现；
- paged KV cache 能还原 dense KV；
- paged KV cache append 后仍能还原和执行 attention；
- paged attention 对齐非 paged 的 quantized attention；
- Triton fused/paged kernel 在 CUDA 上对齐 reference（无 GPU 时自动 skip）。

共 11 个用例。在纯 CPU 环境下 8 个运行、3 个 CUDA/Triton 用例自动 skip：

```text
Ran 11 tests in ...s
OK (skipped=3)
```

在 CUDA + Triton 环境（如 Colab A100）下 3 个 skip 用例会真正运行，验证 Triton kernel 输出与 dequant reference 对齐，并断言 `materializes_dense_kv=false`。

## A100 Microbenchmark

脚本默认 `--device auto --dtype auto`：检测到 CUDA 时自动使用 `cuda + float16`，并在计时前后执行 `torch.cuda.synchronize()`。

Dense baseline：

```bash
python benchmarks/microbench.py --backend dense --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --iters 50 --warmup 10 --repeats 20 --json
```

Kernel 1 路径：

```bash
python benchmarks/microbench.py --backend fused --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --iters 50 --warmup 10 --repeats 20 --json
```

Kernel 1 Triton fused 路径：

```bash
python benchmarks/microbench.py --backend triton_fused --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --iters 50 --warmup 10 --repeats 20 --json
```

Kernel 2 路径：

```bash
python benchmarks/microbench.py --backend paged --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --block-size 16 --iters 50 --warmup 10 --repeats 20 --json
```

Kernel 2 Triton paged 路径：

```bash
python benchmarks/microbench.py --backend triton_paged --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --block-size 16 --iters 50 --warmup 10 --repeats 20 --json
```

保存 JSON：

```bash
python benchmarks/microbench.py --backend triton_paged --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --iters 50 --warmup 10 --repeats 20 --json --include-raw --output results/a100_triton_paged.json --raw-output results/a100_triton_paged_raw.json
```

输出字段包括 `latency_ms`、`latency_p50_ms`、`latency_p90_ms`、`tokens_per_second`、估算 KV 字节数、压缩比、`materializes_dense_kv`、`estimated_effective_*_bandwidth_gbps`、`measured_*` profiler 占位字段、`nsight_compute_command` 和 `latency_breakdown`。`latency_breakdown` 给出 KV load、dequant、QK、softmax、PV accumulation、output write 的阶段定义和粗粒度工作量估算。

## Batch × Seq Len Sweep

```bash
python benchmarks/sweep.py --backends dense,triton_fused,triton_paged --batches 1,4,8,16 --seq-lens 512,1024,2048,4096 --heads 32 --head-dim 128 --iters 20 --warmup 5 --repeats 10 --output results/a100_sweep.csv
```

这个 CSV 可以直接用于画 Pareto 曲线：batch size、sequence length、tokens/s、p50/p90 latency、估算 bandwidth。

## 端到端 Serving 模拟

```bash
python benchmarks/e2e_serving.py --requests 32 --prompt-len 1024 --decode-steps 64 --heads 32 --head-dim 128 --json
```

这不是模型质量 benchmark，而是系统路径 benchmark。它会构建初始 paged quant-KV cache，并在 decode loop 中通过 `append` 追加新 token，输出 TTFT、TPOT 和 tokens/s。

## Roofline SVG

生成 A100 版本 roofline 草图：

```bash
python scripts/profile_roofline.py --peak-tflops 312 --bandwidth-gbps 1555 --intensity 1.0 --achieved-tflops 1.5 --output results/a100_roofline.svg
```

脚本会生成自包含 SVG，不依赖 matplotlib。

## 项目结构

```text
src/flashspec/
  attention.py       dense / quantized decode attention API
  quant.py           per-block affine INT8 量化
  paged.py           paged quant-KV cache 和 block_table
  runtime.py         auto device / dtype / CUDA synchronize
  serving.py         最小 decode serving loop
  triton_utils.py    Triton 可选导入和共享工具
  triton_fused.py    Kernel 1：INT8 KV fused dequant attention
  triton_paged.py    Kernel 2：paged INT8 KV attention
  triton_kernels.py  Triton 兼容导出入口
benchmarks/
  microbench.py      kernel 级 latency 和 bandwidth 估算
  e2e_serving.py     decode loop benchmark
  sweep.py           batch × seq len 扫描
scripts/
  profile_report.py
  profile_roofline.py
tests/
  test_flashspec.py
```

## 更多文档

- `doc/README.MD`：设计背景、当前实现边界和推荐开发顺序。
- `doc/TODO.MD`：按优先级拆分的后续工程任务、验收标准和里程碑。
- `doc/deep_engineering_project.svg`：项目技术地图。

## 在 A100 上做真实 Profiling

建议先用 microbenchmark 固定 shape；JSON 会给出一条可复制的 `nsight_compute_command`。运行该命令后，再把 Nsight Compute 中的 DRAM bytes、achieved bandwidth、occupancy、SM utilization 回填到 `measured_*` 字段或报告中。

示例：

```bash
python benchmarks/microbench.py --backend triton_paged --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --iters 50 --warmup 10 --repeats 20 --device cuda --dtype float16 --json --output results/a100_triton_paged.json
python scripts/profile_report.py results/a100_triton_paged.json --output results/a100_triton_paged_profile.md
```

重点记录：

- KV cache 读取字节数；
- achieved memory bandwidth；
- arithmetic intensity；
- KV load / QK matmul / softmax / value accumulation 的 latency breakdown；
- batch size 和 sequence length sweep 下的 TPOT、tokens/s；
- dense baseline、fused dequant、paged quant-KV 三条路径对比。
