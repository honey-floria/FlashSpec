# FlashSpec TODO

这个 TODO 聚焦后续真正能提升项目质量和技术含金量的工作。优先级含义：

- P0：项目核心价值闭环，没有这些只能算 correctness scaffold。
- P1：让 benchmark、serving 和工程实现更可信。
- P2：进一步优化、扩展和打磨。

## P0：核心 Kernel 闭环

### 1. 实现 Kernel 1：INT8 KV fused dequant attention

当前 `src/flashspec/triton_kernels.py` 只是 wrapper，`fused_dequant_attention` 会先 materialize dense KV。需要实现真正的 Triton/CUDA kernel：

- 从 INT8 K/V block 直接 load；
- 在寄存器或 shared memory 中完成 dequant；
- 完成 QK、softmax、PV；
- 输出 `[batch, heads, head_dim]`；
- 主路径不落地 dense FP16/FP32 KV。

验收标准：

- `materializes_dense_kv=false` 的 fused backend 可运行；
- 输出与 PyTorch reference 在合理误差内对齐；
- 在 A100 上至少覆盖 `head_dim=64/128`、`seq_len=512/2048/4096`；
- benchmark 能对比 dense、portable fused、Triton fused 三条路径。

### 2. [x] 实现 Kernel 2：paged INT8 KV attention

已在 `src/flashspec/triton_kernels.py` 中实现 `paged_quant_attention_triton`：

- 支持 `block_table[b, logical_block] -> physical_block`；
- 支持非整除 `seq_len` 和最后一个 partial block；
- 支持 batch 内不同 request length；
- 避免 gather 后形成完整 dense KV。

验收标准：

- `triton_paged` backend 不调用 `cache.to_dense()`；
- variable length case 与 reference attention 对齐；
- `materializes_dense_kv=false` 的 paged Triton backend 可运行；
- Nsight 中能看到实际读取的是 INT8 KV block，而不是 dense KV tensor。这个 profiling 证据仍需要在 P1 profiling 任务中补充。

## P1：Profiling 和 Benchmark 可信化

### 3. [x] 增加真实 CUDA timing 和 profiler 指标

已在 `benchmarks/microbench.py` 和 `benchmarks/sweep.py` 中把 benchmark 结果拆成“估算字段”和“实测字段”：

- CUDA 路径使用 `torch.cuda.Event` 记录 GPU stream 上的实际耗时，CPU 路径继续使用 `perf_counter`；
- 增加 `--warmup`、`--repeats`，输出 `latency_mean_ms`、`latency_std_ms`、`latency_p50_ms`、`latency_p90_ms`、`latency_p99_ms`；
- `--include-raw` 可把 raw latency 写入 JSON，`--raw-output` 可单独保存 raw measurements；
- JSON 内写入可复制的 `nsight_compute_command`，用于在 A100/Colab 等 CUDA 环境采集 profiler；
- 新增 `estimated_*` bandwidth 字段和 `measured_*` profiler 字段，`dram bytes`、`achieved bandwidth`、`occupancy`、`SM utilization` 当前先置为 `null`，需要运行 Nsight Compute 后回填。

验收标准：

- [x] JSON 输出区分 `estimated_*` 和 `measured_*`；
- [x] sweep CSV 可直接画 latency/tokens/s/bandwidth 曲线；
- [x] README 中有可复现实验命令和 profiling report 生成命令。

### 4. [x] 完善 latency breakdown

已在 microbenchmark JSON 中新增 `latency_breakdown`，用于解释性能主要消耗在哪些阶段，而不只是给一个总耗时：

- KV load；
- dequant；
- QK；
- softmax；
- PV accumulation；
- output write。

每个阶段都会输出：

- `stage`：阶段名称；
- `estimated_bytes`：该阶段粗粒度字节访问估算；
- `estimated_flops_or_ops`：该阶段粗粒度算术或操作量估算；
- `measurement`：建议用 Nsight Compute 采集的指标类型；
- `notes`：中文说明，包括 portable quant backend 会物化 dense KV，而 Triton quant backend 不物化 dense KV。

同时新增 `scripts/profile_report.py`，可以把 microbenchmark JSON 转成 Markdown profiling report，便于查看 `nsight_compute_command`、阶段 breakdown 和 profiler 字段。

验收标准：

- [x] profiler 报告能对应到上述阶段；
- [x] benchmark 文档说明每个阶段的瓶颈判断；
- [x] 当前阶段 breakdown 是“阶段定义 + 工作量估算 + profiler 映射”，真实 per-stage latency 仍需要运行 Nsight Compute 的 source-line/instruction/memory metrics 后归因。

## P1：Serving 模拟接近真实系统

### 5. 扩展 PagedKVCache allocator

当前 `append` 已避免每步从完整 dense KV 重建 cache，但 portable 实现仍会重新量化物理 block store。后续需要：

- block free list；
- request 到 block list 的生命周期管理；
- append token 时只更新目标 block；
- request 完成后释放 block；
- 支持 batch 内 request 新增和结束。

验收标准：

- serving loop 支持不同请求长度；
- decode step 不重新量化全部 physical blocks；
- 可以统计 block utilization 和 fragmentation。

### 6. 区分 prefill 和 decode

真实 serving 中 prefill 和 decode 的计算形态不同：

- prefill：批量写入 prompt KV；
- decode：每 request 每步追加一个 token；
- benchmark 输出 TTFT、TPOT、tokens/s；
- 支持 prompt length 分布，而不是单一固定长度。

验收标准：

- `e2e_serving.py` 支持 prompt length list 或分布参数；
- 输出 prefill time、decode time、TPOT、block utilization。

## P1：测试补强

### 7. 扩大 correctness 测试矩阵

需要覆盖更多真实边界：

- CUDA；
- FP16/BF16；
- `block_size=8/16/32`；
- 非整除 seq len；
- variable lengths；
- append 跨 block 边界；
- batch 内不同 lengths；
- empty/invalid shape 的错误路径。

验收标准：

- CPU 测试默认可跑；
- CUDA 测试在无 GPU 环境自动 skip；
- Triton kernel 测试与 portable reference 共用 case。

### 8. 增加 benchmark schema 测试

benchmark 结果未来会用于画图和对比，需要稳定字段：

- microbench JSON schema；
- sweep CSV fieldnames；
- e2e serving JSON schema；
- `materializes_dense_kv`、`bandwidth_fields_are_estimates` 等边界字段必须存在。

验收标准：

- CI 中运行小规模 benchmark smoke test；
- schema 变更需要同步 README 和 doc。

## P2：量化策略和数值质量

### 9. 评估不同量化粒度

当前是 per-block affine INT8。可继续比较：

- per-head scale；
- per-channel scale；
- symmetric INT8；
- FP8；
- K/V 分别使用不同粒度；
- group-wise scale。

验收标准：

- 增加误差评估脚本；
- 输出 attention output error、top-k attention drift、latency 和 storage cost；
- 给出推荐默认量化配置。

### 10. 加入真实模型 KV 分布评估

随机 tensor 只能验证流程，不能代表真实 LLM KV 分布。后续可以接入真实模型采样：

- 保存真实 prompt 的 KV cache sample；
- 比较不同 layer/head 的 quantization error；
- 分析长上下文下误差变化。

验收标准：

- 至少支持一个开源小模型的 KV sample workflow；
- 结果文档说明误差分布和风险。

## P2：工程体验

### 11. 文档和命令整理

- 根 README 保持快速开始；
- `doc/README.MD` 维护设计说明；
- `doc/TODO.MD` 维护任务路线；
- benchmark 示例输出放入 `results/` 或文档片段。

验收标准：

- 新用户能在 10 分钟内跑通测试和一个小 benchmark；
- 文档明确区分 portable backend 和真实 kernel backend。

### 12. CI 和格式化

- 增加 lint/format 配置；
- 增加 CPU unit test CI；
- CUDA/Triton 测试作为可选 job；
- 避免 benchmark 结果、IDE 文件进入版本控制。

验收标准：

- PR 前可一键运行 `python -m unittest discover -s tests`；
- CI 能发现 API/schema 破坏。

## 建议里程碑

### Milestone 1：可信 scaffold

- 完成 benchmark schema 测试；
- 完成 variable length 和 append 边界测试；
- 文档明确实现边界。

### Milestone 2：Kernel 1 可用

- Triton fused dequant attention 可运行；
- 与 reference 对齐；
- A100 上完成 dense vs fused 对比。

### Milestone 3：Kernel 2 可用

- paged Triton attention 支持 `block_table`；
- 支持 variable lengths；
- 证明不 materialize dense KV。

### Milestone 4：Serving benchmark

- 支持 prefill/decode 分离；
- 支持 request 生命周期；
- 输出可复现实验报告。
