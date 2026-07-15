# FlashSpec TODO

这个 TODO 只保留下一阶段真正需要推进的工作。已经完成的历史方案和实验过程放在 `doc/optimization-log.md`，项目入口说明放在根目录 `README.md`。

## 当前状态

### 已完成

- [x] Kernel 1: `triton_fused` INT8 KV fused dequant attention。
- [x] Kernel 2: `triton_paged` paged INT8 KV attention，支持 `block_table` 和 variable lengths。
- [x] portable reference backend，用于 CPU/CUDA correctness。
- [x] CUDA/Triton correctness tests，无 GPU 时自动 skip。
- [x] `microbench.py` 支持 CUDA event timing、raw latency、`estimated_*` 和 `measured_*` profiler 字段。
- [x] `--profile-ncu` 可直接采集 Nsight Compute fast metrics。
- [x] `scripts/profile_matrix.py` 支持 fused/paged 参数矩阵。
- [x] Colab A100 full matrix + NCU 已完成，结果记录在 `doc/optimization-log.md`。
- [x] `run.ipynb` 已整理为 Colab A100 runbook。

### 当前事实结论

- 推荐默认候选：`block_n=128, num_warps=4`。
- `triton_fused` 推荐 Split-K 候选：`num_splits=4`。
- `block_n=32` 和 `num_warps=8` 当前都不适合作为默认。
- paged uniform 相对 fused 最优仍有约 `12-15%` latency gap，是下一阶段主要优化空间。

## P0：收口当前 kernel/profiling 结论

### 1. 固化默认 kernel 参数

把 full matrix 结论反映到代码默认值和文档中。

- [x] `triton_fused` 默认使用 `block_n=128`、`num_warps=4`、`num_splits=4`。
- [x] `triton_paged` 默认使用 `block_n=128`、`num_warps=4`。
- [x] 保留 `FLASHSPEC_BLOCK_N`、`FLASHSPEC_NUM_WARPS`、`FLASHSPEC_NUM_SPLITS` override。
- [ ] A100 单点复跑确认默认值和 manifest 最优点一致。

验收标准：

- [x] 默认参数 helper 与预期默认参数一致。
- [ ] CUDA correctness 测试通过。
- [ ] A100 sanity profiling 不退化。

### 2. 增加 matrix manifest 自动分析脚本

现在 full matrix 结论主要靠手动读 CSV。需要脚本化，保证以后上传 manifest 后可以直接生成结论。

- [x] 新增 `scripts/analyze_matrix.py`。
- [x] 输出每个 backend/shape 的 top-k latency。
- [x] 输出 `block_n`、`num_warps`、`num_splits`、`paged_layout` 对比。
- [x] 输出 Markdown summary。
- [ ] 可选：生成 heatmap 或 CSV summary。

验收标准：

- [x] 输入 `results/profile_matrix` 后自动生成一份稳定报告。
- [x] 报告能复现当前 full matrix 的主要结论。

### 3. 做 source-line / instruction 归因

当前已经知道最优参数，但还需要解释 paged gap、`block_n=32` 变慢和 `num_warps=8` 变慢的具体来源。

优先点：

- [x] fused 最优：`s2048/d128, block_n=128, num_warps=4, split=4`。
- [ ] fused 最优：`s4096/d128, block_n=128, num_warps=4, split=4`。
- [x] paged uniform 最优：`block_n=128, num_warps=4, contiguous`。
- [x] paged shuffled 慢点。
- [x] `block_n=32` 慢点。
- [x] `num_warps=8` 慢点。

关注指标：

- address calculation / integer instruction overhead；
- dequant 指令占比；
- long scoreboard stall；
- memory workload / global load efficiency；
- register pressure 和 spill。

验收标准：

- [x] `optimization-log.md` 中补一节 source-line 归因结论。
- [x] 能解释 paged 相对 fused 的 `12-15%` gap 主要来自哪里。

## P1：Serving 模拟可信化

### 4. PagedKVCache allocator

当前 serving loop 只能证明 paged path 可以跑通，还不是真实 allocator。

- [ ] block free list。
- [ ] request -> block list 生命周期管理。
- [ ] append token 时只更新目标 block。
- [ ] request 完成后释放 block。
- [ ] 支持 batch 内 request 新增和结束。
- [ ] 输出 block utilization 和 fragmentation。

验收标准：

- decode step 不重建完整 cache。
- benchmark 能报告 utilization / fragmentation。

### 5. 区分 prefill 和 decode

- [ ] `e2e_serving.py` 支持 prompt length list 或分布参数。
- [ ] 输出 prefill time、decode time、TTFT、TPOT、tokens/s。
- [ ] 支持 variable prompt length。

验收标准：

- serving benchmark 更接近真实请求流。
- 结果可以和 paged cache allocator 指标关联。

## P1：测试和工程质量

### 6. 扩大 correctness 测试矩阵

- [ ] CUDA shape: `head_dim={64,128}`、`seq_len={512,2048,4096}`。
- [ ] `block_size={8,16,32}`。
- [ ] 非整除 seq_len。
- [ ] variable lengths。
- [ ] append 跨 block 边界。
- [ ] batch 内不同 lengths。
- [ ] invalid shape / error path。

### 7. Benchmark schema 测试

- [x] microbench JSON smoke test。
- [x] microbench JSON 顶层保留 `num_splits`、`block_n`、`num_warps`、`length_pattern`、`paged_layout`。
- [x] `analyze_results.py` 导出 NCU 诊断列。
- [ ] sweep CSV fieldnames schema test。
- [ ] e2e serving JSON schema test。
- [ ] matrix manifest schema test。

### 8. CI 和格式化

- [ ] 增加 CPU unit test CI。
- [ ] CUDA/Triton 测试作为可选 job。
- [ ] 增加 lint/format 配置。
- [ ] 避免 benchmark JSON、IDE 文件和 `__pycache__` 进入版本控制。

## P2：量化和真实模型评估

### 9. 量化策略对比

当前是 per-block affine INT8。后续可以比较：

- [ ] per-head scale。
- [ ] per-channel scale。
- [ ] symmetric INT8。
- [ ] FP8。
- [ ] K/V 不同量化粒度。
- [ ] group-wise scale。

验收标准：

- 输出 attention output error、top-k attention drift、latency 和 storage cost。
- 给出推荐默认量化配置。

### 10. 真实模型 KV 分布

随机 tensor 只能验证工程流程，不能代表真实 LLM KV 分布。

- [ ] 支持至少一个开源小模型的 KV sample workflow。
- [ ] 保存真实 prompt 的 KV cache sample。
- [ ] 比较不同 layer/head 的 quantization error。
- [ ] 分析长上下文下误差变化。

验收标准：

- 文档说明真实 KV 分布下的误差和风险。

## 文档维护规则

- 根目录 `README.md` 是唯一项目入口。
- `doc/optimization-log.md` 记录实验历史、数据和结论。
- `doc/TODO.MD` 只记录下一步任务。
- 不再维护单独的 `doc/README.MD` 或历史方案文档。
