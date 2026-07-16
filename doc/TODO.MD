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
- [x] `scripts/analyze_matrix.py` 可从 matrix manifest 和 source attribution export 生成稳定 Markdown 报告。
- [x] Colab A100 full matrix + NCU 已完成，结果记录在 `doc/optimization-log.md` 和 `results/profile_matrix_report.md`。
- [x] A100 默认候选已复跑，默认方向仍为 `block_n=128, num_warps=4`，fused 使用 `num_splits=4`。
- [x] `run.ipynb` 已整理为 Colab A100 runbook。

### 当前事实结论

- 当前 matrix 结果仍以 `results/profile_matrix/*/*_manifest.csv` 为准，共 120 个点。
- `triton_fused` 最优点：
  - s2048/d128: `block_n=128, num_warps=4, split=4`，latency 约 `0.2555 ms`。
  - s4096/d128: `block_n=128, num_warps=4, split=4`，latency 约 `0.4856 ms`。
- `triton_paged` uniform 最优点：
  - s2048/d128: `block_n=128, num_warps=4, contiguous`，latency 约 `0.2858 ms`。
  - s4096/d128: `block_n=128, num_warps=4, contiguous`，latency 约 `0.5604 ms`。
- `block_n=32` 和 `num_warps=8` 仍不适合作为默认：它们提高表面 occupancy，但降低 DRAM throughput 和 issue efficiency。
- paged uniform 相对 fused 最优仍有约 `12-15%` latency gap，是下一阶段 kernel 层主要优化空间。
- `results/ncu_source_attribution/` 已包含 s4096 fused/paged best-point `.ncu-rep`；`results/ncu_source_attribution_export/` 已补齐 s4096 的 CSV/TXT 导出，`profile_matrix_report.md` 的 Source Attribution Summary 已同时汇总 s2048 和 s4096 attribution。
- `results/colab_kernels/analysis/summary.csv` 已按当前 `results/colab_kernels/*.json` 重新生成；这些 JSON 是历史单点样本，不是当前默认配置的正式验收依据。

## P0：结果归档和文档收口

### 1. 补齐 s4096 source attribution 导出

- [x] 在有 `ncu` 的环境中导出 s4096 `.ncu-rep`：
  - `fused_best_s4096_d128_bn128_nw4_split4.ncu-rep`
  - `paged_best_s4096_d128_bn128_nw4_uniform_contiguous.ncu-rep`
- [x] 生成对应 CSV/TXT 到 `results/ncu_source_attribution_export/`。
- [x] 重新运行 `scripts/analyze_matrix.py`，让 `results/profile_matrix_report.md` 的 Source Attribution Summary 包含 s4096。
- [x] 把 s4096 attribution 结论补进 `doc/optimization-log.md`。

验收标准：

- [x] `results/ncu_source_attribution_export/` 中存在 s4096 fused/paged CSV 和 TXT。
- [x] `results/profile_matrix_report.md` Source Attribution Summary 同时包含 s2048 和 s4096。
- [x] `optimization-log.md` 明确说明 s4096 是否验证了 s2048 的归因结论。

### 2. 清理 profiling 结果边界

- [x] 明确 `results/profile_matrix/*/*_manifest.csv` 是正式 matrix 结果来源。
- [x] 检查 `results/colab_kernels/*.json`，旧单点结果保留在原目录并标注文档边界，`summary.csv` 已按当前 JSON 重新生成。
- [x] 重新生成 `results/colab_kernels/analysis/summary.csv`，保证 summary 和实际 JSON 一致。
- [x] 在 README 里说明哪些 `results/` 文件是正式结论，哪些只是历史单点样本。

验收标准：

- [x] `summary.csv` 中的关键行和对应 JSON 文件字段一致。
- [x] README 不再让读者误把旧单点 JSON 当作当前默认结果。
- [x] `git status --short` 保持干净。

## P1：Serving 模拟可信化

### 3. PagedKVCache allocator

当前 `PagedKVCache.append()` 已经不是从完整 dense cache 重建，但仍会反量化并重新量化完整 physical block store。下一步要加入真实 allocator 概念，让 serving loop 能表达 request lifecycle、block 复用和碎片。

- [x] 新增 allocator 数据结构，管理固定容量 physical blocks。
- [x] 实现 block free list。
- [x] 实现 `request_id -> logical block list -> physical block` 生命周期管理。
- [x] append token 时只定位并更新目标 request 的最后一个 block；跨 block 边界时才分配新 block。
- [x] request 完成后释放该 request 占用的 physical blocks。
- [x] 支持 batch 内 request 新增和结束。
- [x] 输出 allocator stats：
  - allocated blocks
  - free blocks
  - live requests
  - used tokens
  - capacity tokens
  - block utilization
  - padding waste / fragmentation

验收标准：

- [x] decode step 不重建完整 cache。
- [x] append 跨 block 边界时只新增必要 block。
- [x] request 结束后 block 能被后续 request 复用。
- [x] benchmark 能报告 utilization / fragmentation。
- [x] CPU correctness 测试覆盖 allocator append、release、reuse。

### 4. Serving benchmark：prefill/decode 分离

- [x] `benchmarks/e2e_serving.py` 支持 prompt length list，例如 `--prompt-lens 512,1024,2048`。
- [x] 支持 prompt length 分布参数，例如 uniform/random/bimodal。
- [x] 明确区分 prefill cache build time 和 decode loop time。
- [x] 输出 TTFT、TPOT、tokens/s。
- [x] 输出 allocator utilization / fragmentation。
- [x] 支持 batch 内不同 request length。
- [x] 支持 request arrival / finish 的简单模拟。

验收标准：

- [x] serving benchmark 不再只是一组固定长度请求的 smoke test。
- [x] JSON 输出能把 latency 指标和 allocator 指标关联起来。
- [x] README 给出一个标准 serving benchmark 命令和示例输出字段。

### 5. Serving JSON schema 测试

- [x] 为 `e2e_serving.py --json` 增加 schema smoke test。
- [x] 检查必需字段：
  - `ttft_ms`
  - `tpot_ms`
  - `tokens_per_second`
  - `prefill_ms`
  - `decode_ms`
  - `block_utilization`
  - `fragmentation`
- [x] 确认 CPU 环境下测试可运行，不依赖 CUDA。

## P1：测试和工程质量

### 6. 扩大 correctness 测试矩阵

- [ ] CUDA shape: `head_dim={64,128}`、`seq_len={512,2048,4096}`。
- [ ] `block_size={8,16,32}`。
- [ ] 非整除 seq_len。
- [ ] variable lengths。
- [ ] append 跨 block 边界。
- [ ] batch 内不同 lengths。
- [ ] allocator release/reuse。
- [ ] invalid shape / error path。

验收标准：

- [ ] CPU reference 测试稳定通过。
- [ ] CUDA/Triton 环境下 kernel correctness 覆盖当前默认参数和主要边界。

### 7. Benchmark schema 和报告测试

- [x] microbench JSON smoke test。
- [x] microbench JSON 顶层保留 `num_splits`、`block_n`、`num_warps`、`length_pattern`、`paged_layout`。
- [x] `analyze_results.py` 导出 NCU 诊断列。
- [ ] sweep CSV fieldnames schema test。
- [ ] matrix manifest schema test。
- [ ] `analyze_matrix.py` output smoke test，避免报告列名或 source summary 退化。
- [ ] `profile_report.py` output smoke test。

### 8. CI 和格式化

- [ ] 增加 CPU unit test CI。
- [ ] CUDA/Triton 测试作为可选 job。
- [ ] 增加 lint/format 配置。
- [ ] 明确哪些 `results/` 产物纳入版本控制。
- [ ] 避免临时 benchmark JSON、IDE 文件和 `__pycache__` 进入版本控制。

## P2：Kernel 后续优化

### 9. Paged path 地址计算优化

source attribution 显示 paged 相对 fused 的主要 gap 是较低 DRAM BW / issue efficiency，而不是 local spill。下一步应针对 block table 和地址计算做小步 A/B。

- [ ] 审查 `triton_paged` 内层循环的 block table load、physical block index 和 quant block offset 计算。
- [ ] 尝试把可复用地址计算移出内层循环。
- [ ] 对 `page_block_size == quant_block_size` 的快路径继续做专门化，减少恒等除法/乘法。
- [ ] 每个 patch 只跑小矩阵 + source attribution，不再盲目跑 full matrix。

验收标准：

- [ ] s2048/s4096 paged uniform 相对 fused 的 gap 有可测下降。
- [ ] 不牺牲 correctness 和 variable length 支持。

### 10. MIO / short scoreboard 来源定位

- [ ] 对 `block_n=128,nw4` 主路径检查 shared staging、dequant、softmax update 的指令和依赖链。
- [ ] 避免回到单纯追 occupancy 的方向。
- [ ] 若尝试降寄存器，必须同时检查 DRAM throughput、issue/cycle、MIO 和 scoreboard，而不是只看 regs/occupancy。

验收标准：

- [ ] 每个 kernel patch 都有明确的 profiling 假设、命令、结果和结论。
- [ ] `optimization-log.md` 继续按“动机 -> 做法 -> 结果 -> 原因分析 -> 结论”追加。

### 11. Kernel 3: tree-attn / speculative decode

Kernel 3 目标是实现 speculative decoding 的 tree verify kernel：在 quant/paged KV 上并行验证 draft tree，把 causal mask 改成 tree mask，并支持树状 KV 读取。它是可选 bonus，但如果要做，就按“先正确、再可测、再提速”的顺序推进。

- [ ] 明确 Kernel 3 的输入/输出约定：
  - 输入：`q`、paged/quant KV、tree 拓扑、每个节点的 parent / depth / path token 信息。
  - 输出：每个 draft 节点的 verify 结果、accept/reject 标记，或每个节点的 attention 输出。
- [ ] 设计 tree 表示层：
  - 用 compact node table 表示树节点；
  - 用 parent pointer / path index 组织祖先链；
  - 明确哪些节点共享前缀 KV，哪些节点需要独立读取。
- [ ] 先做 CPU reference：
  - 把 tree mask 显式展开成 dense mask；
  - 用现有 reference/paged attention 验证 tree verify 的数学正确性；
  - 覆盖单链、分叉、不同深度、不同 branch factor。
- [ ] 复用 Kernel 2 的 paged KV 基础：
  - 先确认 tree-attn 读取仍基于 block_table / quant block store；
  - 再在读取阶段加入 path-aware gather，而不是先改 allocator。
- [ ] 设计 tree mask kernel 逻辑：
  - 只让当前节点看到祖先路径上的 token；
  - 屏蔽兄弟分支和未来节点；
  - 保证和 speculative verify 的 acceptance 语义一致。
- [ ] 实现树状 gather / verify 主 kernel：
  - 优先正确实现路径读取；
  - 再考虑是否能把多节点合并成一个 program；
  - 关注 branch divergence、地址计算和不规则访存。
- [ ] 增加必要的辅助数据结构：
  - node -> path mapping；
  - path prefix length；
  - subtree/leaf 采样信息；
  - 需要时缓存祖先 token 的 block 索引。
- [ ] 增加 correctness 测试：
  - 单路径退化时应等价于 Kernel 2；
  - 两层/三层 tree 的 verify 结果应与 dense tree mask 对齐；
  - 不同 branch factor 和不同深度都要覆盖；
  - invalid tree / cycle / orphan node 必须报错。
- [ ] 增加 benchmark / profiling 入口：
  - 固定小树形状做 smoke test；
  - 变 branch factor / depth / leaf count 做 microbench；
  - 采集 NCU 归因，重点看 address 计算、long scoreboard、global load efficiency。
- [ ] 明确和现有 Kernel 1/2 的边界：
  - Kernel 1 仍负责 fused linear decode；
  - Kernel 2 仍负责 linear paged decode；
  - Kernel 3 只负责 tree verify 场景，不把 speculative logic 混进通用 decode path。

验收标准：

- [ ] tree-attn 在 CPU reference 上和 dense tree mask 完全对齐。
- [ ] 单路径退化时，Kernel 3 行为与现有 paged decode 一致。
- [ ] 分叉 tree 能正确 verify，accept/reject 结果稳定。
- [ ] benchmark / JSON / profiling 输出能区分 tree depth、branch factor、leaf count。
- [ ] `optimization-log.md` 能记录 tree-attn 的动机、实现、结果和瓶颈。

## P2：量化和真实模型评估

### 12. 量化策略对比

当前是 per-block affine INT8。后续可以比较：

- [ ] per-head scale。
- [ ] per-channel scale。
- [ ] symmetric INT8。
- [ ] FP8。
- [ ] K/V 不同量化粒度。
- [ ] group-wise scale。

验收标准：

- [ ] 输出 attention output error、top-k attention drift、latency 和 storage cost。
- [ ] 给出推荐默认量化配置。

### 13. 真实模型 KV 分布

随机 tensor 只能验证工程流程，不能代表真实 LLM KV 分布。

- [ ] 支持至少一个开源小模型的 KV sample workflow。
- [ ] 保存真实 prompt 的 KV cache sample。
- [ ] 比较不同 layer/head 的 quantization error。
- [ ] 分析长上下文下误差变化。

验收标准：

- [ ] 文档说明真实 KV 分布下的误差和风险。

## 文档维护规则

- 根目录 `README.md` 是唯一项目入口。
- `doc/optimization-log.md` 记录实验历史、数据和结论。
- `doc/TODO.MD` 只记录下一步任务。
- 不再维护单独的 `doc/README.MD` 或历史方案文档。
