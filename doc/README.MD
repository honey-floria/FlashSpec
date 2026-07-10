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
- microbenchmark、sweep 和 roofline SVG 脚本。

需要注意：`fused` 是 portable PyTorch 路径，仍会反量化并还原 dense KV，再调用 reference attention；`triton_fused` 是 Kernel 1 的 Triton 路径，主路径不 materialize dense KV。`paged` 是 portable PyTorch 路径，仍会通过 `cache.to_dense()` 还原 KV；`triton_paged` 是 Kernel 2 的 Triton 路径，主路径不 materialize dense KV。

## 关键边界

- benchmark 中的 bandwidth 字段是估算值，不是 Nsight Compute 实测值。
- `materializes_dense_kv=true` 表示该路径仍会创建 dense KV 中间结果。
- `triton_kernels.py` 已包含 Kernel 1 fused dequant attention 和 Kernel 2 paged INT8 KV attention 的 Triton 实现。
- serving 模拟已经支持 cache append，但 portable append 仍会重新量化物理 block store；真实 serving 还需要 allocator/free list 和 request 生命周期管理。

## 推荐开发顺序

1. 先完成 Kernel 1：非 paged INT8 KV fused dequant attention。
2. 再完成 Kernel 2：加入 `block_table` 的 paged INT8 KV attention。
3. 用 CUDA event 和 Nsight Compute 建立可信 profiling。
4. 扩展 serving 模拟，支持 variable length、block allocator、request 完成和新增。
5. 补齐 CUDA、FP16/BF16、边界 shape 和 benchmark schema 测试。

详细任务见 `TODO.MD`。
