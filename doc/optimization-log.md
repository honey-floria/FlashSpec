# FlashSpec 内核优化实验日志

记录 Triton decode attention 内核(Kernel 1 fused / Kernel 2 paged)的每一次优化尝试:
**动机 → 做法 → 结果 → 原因分析 → 结论**。每次新尝试都往下追加,不删旧记录(失败的尝试同样有价值)。

## 基线环境
- GPU: A100 (108 个 SM,每 SM 最多 64 warp)
- 固定配置: batch=16, heads=32, block_size=16, dtype=fp16
- 基准 shape: head_dim=64/128 × seq_len=512/2048/4096
- 指标来源: Nsight Compute (ncu),按 kernel 名过滤,只 profile attention 主 kernel
- 起点瓶颈: occupancy ~25%, DRAM throughput ~44%(内存瓶颈型 kernel,但带宽没打满)

---

## 实验 1: num_warps 4 → 8 ❌ 失败(全面变慢)

**动机**
occupancy 卡在 ~25%。初始诊断:grid 只有 `batch*heads=512` 个 program,撒到 108 个 SM 上每个只落 ~4.7 个,
每 program 4 warp → 每 SM ~19 warp(满编 64)→ occupancy ~25%。推测"给每个 program 多分 warp 能提占用率"。

**做法**
`triton_fused.py` / `triton_paged.py` 的 kernel launch `num_warps` 从 4 改成 8。一行改动,先验证方向。

**结果(A100 实测,latency ms)**

| shape | fused 4→8 | paged 4→8 |
|---|---|---|
| s512/d64 | 0.064→0.095 (+48%) | 0.067→0.085 (+27%) |
| s2048/d128 | 0.361→0.486 (+35%) | 0.331→0.581 (+76%) |
| s4096/d128 | 0.600→0.955 (+59%) | 0.649→1.156 (+78%) |

**每个 shape 都变慢 27–78%,方向完全错误。**

**原因分析**
每个 program 的活太小(每轮只处理 block_n=64 × block_d 一小块),且 QK/softmax/PV 都要跨 warp 归约(`tl.sum`)。
4→8 warp 后,同一小块活被劈成两半,每 warp 只干一半,但"8 个 warp 凑一起归约"的同步 + 共享内存开销几乎翻倍。
**总活量没变,协调开销翻倍 → 净变慢。** 加 warp 让账面 occupancy 升了,但升上来的 warp 在抢同一块活、不是做独立的活,不值钱。

**结论**
- 已回退到 `num_warps=4`。
- 否掉了"加 warp 提占用率"这条便宜路。
- 但证实核心诊断:512 个 program 太少。正解应是**增加独立任务数**(而非给现有任务塞 warp)→ 引出实验 2 (Split-K)。

---

## 实验 2: Split-K (Flash-Decoding) ⚠️ 部分成功(小提速,但机制与预期不符)

**动机**
实验 1 指向"独立任务数不够"。decode attention 天然只有 `batch*heads=512` 个独立任务(query 只有 1 个 token)。
唯一能再切出独立任务的维度是序列长度 → 把"扫 seq_len"沿序列切成 S 段并行,grid 从 512 → 512×S,每段活量不变。

**做法**
- 新增 `_fused_dequant_attention_split_kernel`(grid=(batch*heads, S),每段只扫自己那截,输出 partial online-softmax 状态到 scratch)。
- 新增 `_combine_splits_kernel`(grid=(batch*heads,),跨段合并 partial 状态,数学与 kernel 内跨 block 的 rescale 同构)。
- 自适应 S = clamp(ceil(seq_len/512), 1, 32):s512→1, s2048→4, s4096→8。
- S==1 快路径:直接走原单 kernel,不分配 scratch、不启 combine。
- 环境变量 `FLASHSPEC_NUM_SPLITS` 作 A/B 开关(=1 强制关,隔离 Split-K 纯贡献)。
- 加了 `seq_len=1500` 强制 S>1 的 CUDA correctness 测试,含 variable lengths 覆盖尾段 mask。

**结果(A100 实测,A/B OFF=S1 vs ON=自适应)**

| seq | OFF ms | ON ms | speedup | occ OFF | occ ON | DRAM% | 带宽 GB/s |
|---|---|---|---|---|---|---|---|
| 2048 | 0.3807 | 0.3700 | +2.9% | 24.9% | 23.5% | 47→50 | 738→773 |
| 4096 | 0.7482 | 0.7120 | +4.8% | 25.0% | 24.2% | 47→51 | 738→786 |

延迟小幅下降(2.9–4.8%,已含 combine 开销仍净赚),**但占用率没升,反而略降**。

**原因分析(重要:推翻了实验 1 的核心假设)**
- 占用率 **不是**由 block 总数决定,而是由**每个 block 吃多少寄存器/共享内存**决定。
- 现状:每个 SM 因寄存器压力最多同时容纳 ~4 个 block(4×4warp=16warp=25% occupancy)。即使 Split-K 把 block 从 512 加到 4096,
  每 SM 还是只装 ~4 个,多出来的 block 只能排队分批(wave)上 → per-SM warp 数不变 → occupancy 不变。
- 那 3-5% 提速来自**尾部效应减轻 / 负载更均衡**:block 切小切多后收尾颗粒更细,拖尾更短,DRAM throughput 47%→50%。不是占用率的功劳。

**结论**
- Split-K 实现正确(数学、边界、内存布局均静态审查通过,测试覆盖 S>1)。
- 收益仅 3-5%,考虑到多了 2 个 kernel + scratch 显存,**性价比一般**。保留(自适应,长序列才启用),但不是主要杠杆。
- **真正瓶颈是每个 program 的资源占用(大概率寄存器),不是并行度。** 要大幅提速需降寄存器 → 引出实验 3(诊断)。

---

## 实验 3: 占用率瓶颈诊断 metric 🔬 进行中(待 Colab 数据)

**动机**
实验 2 证明"加 block 无法提 occupancy",最可能的解释是寄存器受限,但缺直接证据(此前只采集 6 个 metric,无寄存器数)。

**做法**
- `microbench.py` NCU 采集新增 `launch__registers_per_thread`(每线程寄存器)和
  `sm__maximum_warps_per_active_cycle_pct`(理论占用率上限)。
- `ncu_parse.py` 解析并回填 `measured_registers_per_thread` / `measured_theoretical_occupancy_pct`,
  取耗时最长的主 kernel 值(避免被 combine 小 kernel 稀释)。
- run.ipynb A/B 对比表加 `reg` / `theoOcc` 两列。

**结果(A100 实测)**

| | reg/thread | 理论占用率 | 实测占用率 | 实测/理论 | bw GB/s |
|---|---|---|---|---|---|
| OFF (单 kernel) | 96 | 31.25% | 24.9% | 80% | 739 |
| ON (split kernel) | 114 | 25.00% | 23.5% | 94% | 776 |

**原因分析(确认寄存器受限,算术精确吻合)**
理论占用率正好等于寄存器算出来的上限:
```
A100: 每 SM 65536 寄存器, 最多 64 warp; num_warps=4 → 128 thread/block
单 kernel: 96 reg × 128 = 12288 reg/block → 65536/12288 = 5.33 → 5 block/SM
          5 block × 4 warp = 20 warp → 20/64 = 31.25%  ← 与 theoOcc 完全一致
split:    114 reg × 128 = 14592 → 65536/14592 = 4.49 → 4 block × 4 = 16 warp = 25%  ← 也吻合
```
- **占用率天花板被寄存器压死**:每线程 96 寄存器 → 每 SM 只能放 5 个 block。加 block 无用(实验 2 已证)。
- **实测 24.9% 已达天花板 31.25% 的 80%**,那 20% 缺口是尾部效应——正是 Split-K 唯一能捞的。Split-K 把比例提到 94%(23.5/25),但同时因多用寄存器把天花板从 31.25% 拉到 25%,两效应抵消 → 解释了实验 2"占用率没升但延迟略降"。

**结论**
- 寄存器受限确认无疑。**唯一能抬高天花板的旋钮是降低每线程寄存器数(< 96)。**
- 潜力估算:reg 96→64 → 8 block/SM → 50%;96→48 → 10 block → 62.5%。
- kernel 是内存瓶颈型(DRAM 47%,带宽 739 ≈ A100 峰值一半),占用率翻倍有望把带宽推上去 → 引出实验 4(降寄存器)。
- 下一步候选:调小 `block_n`(64→32,减小 k_deq/v_deq 临时 tile);Triton `maxnreg` 编译提示;调 `num_stages`。

---

## 实验 4: 降寄存器 — block_n 64→32 ❌ 当前结果变慢(需复跑确认)

**动机**
实验 3 确认占用率天花板被寄存器压死(96 reg → theoOcc 31.25%)。唯一能抬天花板的旋钮是降每线程寄存器数。
`k_deq`/`v_deq` 临时 tile 是 `[block_n, block_d]`,寄存器占用与 block_n 成正比 → 调小 block_n 是最直接的降寄存器手段。

**做法**
- `triton_fused.py` 新增 `_resolve_block_n()`:环境变量 `FLASHSPEC_BLOCK_N` 覆盖(默认 64,可选 16/32/128,非法回退 64)。
- `block_n` 回填进 stats,供 A/B 表显示。
- run.ipynb 加实验 4 cell:固定 `FLASHSPEC_NUM_SPLITS=1`(单 kernel,和实验 3 基线对齐),只变 block_n∈{64,32},
  对比 reg / theoOcc / occ / dram% / bw / latency。

**判读方法(待填实测)**
- block_n=32 应减小 tile → reg 下降。若 reg 压到 ~64 → theoOcc 应升到 ~50%(65536/(64×128)=8 block/SM)。
- 再看延迟:内存瓶颈型 kernel,占用率升 → 并发访存增多 → 带宽升 → 期望延迟降。
- 代价:block_n 减半 → 循环轮数翻倍,若 reg 没降够或访存效率下降,可能反而变慢。

**结果(A100 实测,固定 NUM_SPLITS=1; 以当前 `results/colab_kernels/bn_*.json` 为准)**

| seq | block_n | reg | theoOcc | occ | dram% | bw | latency | 提速 |
|---|---|---|---|---|---|---|---|---|
| 2048 | 64 | 96 | 31.25% | 24.9% | 47.6 | 740 | 0.3812 | 基线 |
| 2048 | 32 | 80 | 37.50% | 26.0% | 30.5 | 475 | 0.5957 | -36.0% |
| 4096 | 64 | 96 | 31.25% | 24.9% | 47.4 | 738 | 0.7486 | 基线 |
| 4096 | 32 | 80 | 37.50% | 26.2% | 30.5 | 474 | 1.1674 | -35.9% |

**原因分析(机制一半成立,但性能失败)**
`block_n 64→32` 确实把寄存器从 96 降到 80,理论占用率从 31.25% 抬到 37.5%。
算术吻合:80 reg × 128 = 10240 reg/block → 65536/10240 = 6.4 → 6 block/SM → 6×4=24 warp = 37.5%(与 theoOcc 一致)。

但这没有转化成有效带宽:实测 occupancy 只从约 24.9% 到 26%,DRAM throughput 反而从约 47% 掉到约 30%,latency 明显变差。
说明单纯减小 `block_n` 会增加循环轮数、地址计算/控制开销,并可能破坏内存访问或指令调度效率。寄存器降低是真的,但不是净收益。

**结论**
- `block_n=32` 当前不应作为默认优化,除非复跑数据推翻当前 JSON。
- 降寄存器仍是正确目标,但不能只靠缩小扫描 tile。下一步应优先攻不随 `block_n` 变化的寄存器大头(q、acc、softmax 状态、kv_offsets 地址张量),并同时观察是否发生 spill/访存效率下降。
- 新增 benchmark schema 字段:JSON 顶层必须记录 `block_n`、`env_flashspec_num_splits`、`env_flashspec_block_n`,避免后续实验无法反查真实配置。
- 下一步候选:① 修短 `kv_offsets`/qparam 地址张量生命周期;② 试 `num_stages`/`maxnreg` 并检查 spill;③ 做 `block_n={16,32,64,128}` 复跑矩阵;④ 把同样诊断字段和优化旋钮迁移到 `triton_paged`。

---

## 实验 5: Kernel profiling 矩阵与 source-line 归因基础设施 🧪 待 A100 复跑

**动机**
实验 4 证明“occupancy 变高”不能直接等价为“性能变好”。下一步需要从单点 A/B 变成矩阵化 profiling：

- Kernel 1: 同时 sweep `num_splits / block_n / num_warps`，判断 Split-K、tile 和 warp 之间的交互；
- Kernel 2: 单独看 `block_n / num_warps / block_table locality / variable lengths`，不能套用 Kernel 1 的结论；
- Nsight Compute 不只看总带宽/occupancy，还要能进入 source-line / instruction / memory workload 归因。

**做法**

- `triton_fused.py` 新增 `FLASHSPEC_NUM_WARPS` 覆盖，stats 回填 `num_warps`。
- `triton_paged.py` 新增 `FLASHSPEC_BLOCK_N` 和 `FLASHSPEC_NUM_WARPS` 覆盖，Kernel 2 也能做 tile/warp sweep。
- `PagedKVCache.from_dense()` 新增 profiling layout:
  - `contiguous`: 原始连续物理 block；
  - `shuffled`: 随机打乱 physical block，保持逻辑 KV 不变；
  - `interleaved`: 按 logical block 跨 batch 交错，观察 locality 变化。
- `microbench.py` 新增:
  - `--length-pattern {uniform,descending,bimodal,random}` 和 `--lengths`；
  - `--paged-layout` / `--layout-seed`；
  - JSON 字段 `num_warps`、`env_flashspec_num_warps`、`length_pattern`、`effective_min/max_seq_len`、`paged_layout`；
  - `nsight_compute_source_command`，用于 source-line / instruction 归因。
- 新增 `scripts/profile_matrix.py`：
  - Kernel 1 矩阵：`seq_len × head_dim × block_n × num_warps × num_splits × length_pattern`；
  - Kernel 2 矩阵：`seq_len × head_dim × block_n × num_warps × length_pattern × paged_layout`；
  - 可选 `--profile-ncu` 直接回填 fast metrics，并输出 manifest CSV。

**推荐命令**

Kernel 1 先跑小矩阵确认方向：

```bash
python scripts/profile_matrix.py --backend triton_fused \
  --seq-lens 2048,4096 --head-dims 128 \
  --block-ns 32,64,128 --num-warps 4,8 \
  --num-splits auto,1,4,8 \
  --length-patterns uniform \
  --profile-ncu --output-dir results/profile_matrix/fused
```

Kernel 2 单独看 locality 和 variable length：

```bash
python scripts/profile_matrix.py --backend triton_paged \
  --seq-lens 2048,4096 --head-dims 128 \
  --block-ns 32,64,128 --num-warps 4,8 \
  --length-patterns uniform,descending \
  --paged-layouts contiguous,shuffled,interleaved \
  --profile-ncu --output-dir results/profile_matrix/paged
```

单点 source-line / instruction 归因使用 microbench JSON 里的 `nsight_compute_source_command`。它会采集
`SourceCounters / InstructionStats / MemoryWorkloadAnalysis / SchedulerStats`，用于回答：

- dequant 转换和 scale/zero 计算是否贡献大量指令；
- QK/PV 是否主要是普通 scalar/vector 指令，而不是 tensor-core 路径；
- paged load 是否因为 block_table 间接寻址产生 long scoreboard stall；
- `block_n=32` 变慢到底是访存效率下降、循环/地址开销增加，还是 instruction mix 变差。

**当前结论**

这是 profiling 基础设施改造，**还不是性能结论**。下一步必须在 A100 上跑矩阵并更新本日志：

- 先比较 `block_n=64` 与 `block_n=128`，验证是否能在不显著增寄存器的情况下提升 DRAM throughput；
- 对 Kernel 1 看 `num_splits` 与 `num_warps` 是否存在 shape-dependent 最优点；
- 对 Kernel 2 看 `shuffled/interleaved` 是否明显拉高 latency 或 long scoreboard stall；
- 如果 source-line 显示地址计算/反量化指令占比过高，再进入代码级寄存器生命周期和地址张量优化。

**2026-07-15 上传结果初读（`results/colab_kernels/*.json`）**

当前上传的是单点 shape + Split-K A/B + `block_n` A/B，还不是 `scripts/profile_matrix.py` 生成的完整矩阵：

- 没有发现 `results/profile_matrix/*manifest.csv`；
- `results/colab_kernels/analysis/summary.csv` 已重新生成；
- dense baseline 的 NCU CSV 解析失败，但 Triton fused/paged 的 profiler 字段完整。

需要注意：这批 JSON 来自一个中间版本，当 `length_pattern=uniform` 时也会把 `lengths` 传给 Kernel 1，导致 `triton_fused` 编译为 `has_lengths=True` 路径，和旧固定长度基线不完全可比。代码已修正为：默认 uniform 不传 `lengths`，只有显式 `--lengths` 或 variable pattern 才测试 mask 分支。下面数字只能作为方向判断，正式表需要修正后复跑。

方向性观察：

| 实验 | 结果 |
|---|---|
| Split-K s2048/d128 | event latency `0.3214 -> 0.3039 ms`，约 `+5.7%` |
| Split-K s4096/d128 | event latency `0.6284 -> 0.5639 ms`，约 `+11.4%` |
| block_n 64->32 s2048/d128 | `0.3442 -> 0.4534 ms`，约 `-24.1%`；DRAM throughput `45.5% -> 31.7%` |
| block_n 64->32 s4096/d128 | `0.6279 -> 0.8947 ms`，约 `-29.8%`；DRAM throughput `45.3% -> 31.4%` |
| triton_paged vs triton_fused | paged 在 d128/s2048 接近 fused（`0.3706 vs 0.3666 ms`），但多数 shape 仍慢，d64 差距更大 |

临时结论不变：`block_n=32` 不能作为默认；Split-K 值得保留并继续按 `num_splits` 矩阵找最优；Kernel 2 下一步必须跑 `paged_layout={contiguous,shuffled,interleaved}` 和 variable length 矩阵，否则无法判断 block_table locality 是否是主要损耗。

**2026-07-15 small matrix + NCU 复跑结果（`results/profile_matrix/*/*manifest.csv`）**

这次已经跑完 `MATRIX_PRESET=small, PROFILE_NCU=True`：

- `triton_fused`: 16 个点，全部有 `measured_achieved_bandwidth_gbps / DRAM throughput / occupancy / registers`，无 `profiler_error`；
- `triton_paged`: 16 个点，全部有 NCU fast metrics，无 `profiler_error`；
- small 矩阵覆盖 `seq_len={2048,4096}, head_dim=128, block_n={64,128}, num_warps=4`；
- fused 覆盖 `num_splits={auto,1,4,8}`；paged 覆盖 `length_pattern={uniform,descending}, paged_layout={contiguous,shuffled}`。

每组 latency 最优点：

| backend | seq_len | 最优配置 | latency_ms | measured BW | DRAM throughput | occupancy | regs/thread |
|---|---:|---|---:|---:|---:|---:|---:|
| triton_fused | 2048 | `block_n=128, num_splits=8` | 0.2619 | 908.5 GB/s | 58.4% | 18.3% | 168 |
| triton_fused | 4096 | `block_n=128, num_splits=4` | 0.4865 | 929.9 GB/s | 59.8% | 17.9% | 168 |
| triton_paged | 2048 | `block_n=128, descending, contiguous` | 0.2694 | 529.7 GB/s | 34.1% | 15.6% | 168 |
| triton_paged | 4096 | `block_n=128, descending, contiguous` | 0.5275 | 522.0 GB/s | 33.6% | 15.6% | 168 |

`block_n=128` 在 small 矩阵里明确优于 `block_n=64`：

| backend | seq_len | best block_n=64 | best block_n=128 | block_n=128 speedup |
|---|---:|---:|---:|---:|
| triton_fused | 2048 | 0.3026 ms | 0.2619 ms | +15.5% |
| triton_fused | 4096 | 0.5709 ms | 0.4865 ms | +17.3% |
| triton_paged | 2048 | 0.3010 ms | 0.2694 ms | +11.7% |
| triton_paged | 4096 | 0.5914 ms | 0.5275 ms | +12.1% |

这进一步强化实验 4 的结论：当前瓶颈不是“occupancy 数字不够高”。`block_n=128` 寄存器更高、occupancy 更低，但 DRAM throughput 和 latency 更好；应优先保留更大的 scan tile，提高内存效率和减少循环/地址开销。

Split-K 结论：

- `s2048/d128`: 最优是 `num_splits=8`，`0.2619 ms`；`split=4` 为 `0.2881 ms`，`auto` 为 `0.3103 ms`。
- `s4096/d128`: 最优是 `num_splits=4`，`0.4865 ms`；`auto` 为 `0.4872 ms`，几乎相同；`split=8` 为 `0.4948 ms`。
- 因此 Split-K 最优值是 shape-dependent。默认策略可以考虑：`seq_len<=2048` 偏向 8，`seq_len>=4096` 偏向 4 或 auto；正式改默认前还需要 full matrix 或更多 shape 复核。

Paged locality / variable length 观察：

- `descending` 的 event latency 更低，主要因为有效 token 数变少；这不是 uniform shape 的直接优化收益，后续比较必须按 effective token/bytes 一起看。
- `contiguous` 在 descending 下明显优于 shuffled，尤其 s2048: `0.2694 ms` vs `0.3361 ms`，说明 block_table locality 对 variable length/paged 场景有实际影响。
- uniform 下 s2048 的 shuffled 反而快于 contiguous（`0.3028 ms` vs `0.3550 ms`），这可能是单次测量噪声、物理 block 分布副作用，或 cache/set 冲突差异；需要增加 repeats 或加入 `interleaved` 后再定论。

下一步优化方向：

1. 跑 `full + PROFILE_NCU=False`，快速补上 `num_warps=8`、`block_n=32`、paged `interleaved` 的 latency 排名。
2. 对 full latency 前 3-5 个候选再开 `PROFILE_NCU=True`，避免全矩阵 NCU 过慢。
3. fused 默认候选优先测试 `block_n=128`，Split-K 根据 `seq_len` 分段选择。
4. paged 继续重点看 block_table locality：`contiguous/shuffled/interleaved` + source-line，确认 long scoreboard 或地址计算是否是主要损耗。

**2026-07-15 full matrix + NCU 结果（`results/profile_matrix/*/*manifest.csv`）**

新的 manifest 已覆盖 full 矩阵，并且这次 120 个点全部带 NCU fast metrics：

- `triton_fused`: 48 个点，`block_n={32,64,128}`、`num_warps={4,8}`、`num_splits={auto,1,4,8}`；
- `triton_paged`: 72 个点，`block_n={32,64,128}`、`num_warps={4,8}`、`length_pattern={uniform,descending}`、`paged_layout={contiguous,shuffled,interleaved}`；
- 两个 manifest 均无 `profiler_error`，`measured_achieved_bandwidth_gbps` 全部有效。

全矩阵最优点：

| backend | 场景 | 最优配置 | latency_ms | measured BW | DRAM throughput | occupancy | regs/thread |
|---|---|---|---:|---:|---:|---:|---:|
| triton_fused | s2048/d128 | `block_n=128, num_warps=4, split=4` | 0.2555 | 918.9 GB/s | 59.1% | 18.0% | 168 |
| triton_fused | s4096/d128 | `block_n=128, num_warps=4, split=4` | 0.4856 | 931.6 GB/s | 59.9% | 18.0% | 168 |
| triton_paged | s2048/d128 uniform | `block_n=128, num_warps=4, contiguous` | 0.2859 | 790.7 GB/s | 50.8% | 15.5% | 168 |
| triton_paged | s4096/d128 uniform | `block_n=128, num_warps=4, contiguous` | 0.5604 | 789.4 GB/s | 50.8% | 15.5% | 168 |
| triton_paged | s2048/d128 descending | `block_n=128, num_warps=4, contiguous` | 0.2770 | 527.7 GB/s | 33.9% | 15.6% | 168 |
| triton_paged | s4096/d128 descending | `block_n=128, num_warps=4, interleaved` | 0.5289 | 524.0 GB/s | 33.7% | 15.6% | 168 |

关键结论：

- `block_n=128, num_warps=4` 是当前最稳的默认候选。相对同场景最佳 `block_n=64`，fused 提升约 `+16.6%/+17.5%`，paged uniform 提升约 `+16.8%/+15.7%`。
- `block_n=32` 继续失败。虽然 occupancy 更高、寄存器更低，但相对最佳 `block_n=128` 慢约 `49-67%`，DRAM throughput 明显低，说明小 tile 增加循环/地址/调度开销后得不偿失。
- `num_warps=8` 也不适合作为默认。即使寄存器较低、occupancy 看起来更高，最佳 `nw8` 仍比 `bn128,nw4` 慢约 `29-49%`。
- fused 的 Split-K 在 full 矩阵里 `split=4` 对 s2048 和 s4096 都最优；s2048 的 `split=8` 很接近但略慢，后续默认可先固定 `split=4`，再保留 shape-specific tuning 空间。
- paged uniform 相比 fused 最优慢约 `+11.9%`(s2048) 和 `+15.4%`(s4096)。这是 paged/block_table 间接访问的当前成本量级。
- paged locality: s2048 下 `shuffled` 明显变慢，尤其 descending `0.2770 -> 0.3401 ms`；s4096 下三种 layout 接近，说明 locality 损耗更明显出现在较短序列或 cache 更敏感的场景。

更新后的下一步：

1. 把默认实验候选收敛到 `block_n=128, num_warps=4, split=4`，先做 correctness + 单点复跑确认稳定性。
2. 对 fused 最优点和 paged 最优/最差 locality 点跑 source-line / instruction attribution，重点看地址计算、dequant 指令、long scoreboard。
3. 若继续优化寄存器，不再优先靠 `block_n` 或 `num_warps` 降寄存器，而是缩短 q/acc/softmax 状态和 offset/qparam 地址张量生命周期，并检查 spill。
4. 对 paged 优先优化 block_table/locality 路径；uniform 下目标是缩小相对 fused 的 `12-15%` gap。

**2026-07-15 full matrix fast-metrics 归因分析**

当前上传的是 full matrix JSON + NCU fast metrics，尚未包含 `.ncu-rep` 或 source-line/instruction 导出。因此本节是基于
`measured_dram_bytes / achieved bandwidth / DRAM throughput / SM utilization / occupancy / registers` 的归因，不是逐源码行或 SASS 指令归因。

总体相关性：

| backend | latency vs measured BW | latency vs DRAM throughput | latency vs occupancy | 解读 |
|---|---:|---:|---:|---|
| triton_fused | -0.69 | -0.69 | +0.43 | 越能打满 DRAM 越快；occupancy 越高反而常常更慢 |
| triton_paged | -0.57 | -0.57 | +0.53 | 同样主要受访存效率影响，而不是 occupancy 数字 |

这说明当前优化目标应是提高有效访存/调度效率，而不是单纯追 occupancy。`block_n=32` 和 `num_warps=8` 都是典型反例：它们降低寄存器或提高 occupancy，但 DRAM throughput 明显下降。

按参数组平均：

| backend | block_n | num_warps | avg latency | avg BW | avg DRAM throughput | avg occupancy | avg regs |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused | 32 | 4 | 0.5910 ms | 567.6 GB/s | 36.5% | 36.7% | 74 |
| fused | 64 | 4 | 0.4478 ms | 771.8 GB/s | 49.6% | 24.1% | 109.5 |
| fused | 128 | 4 | 0.3880 ms | 900.7 GB/s | 57.9% | 17.5% | 168 |
| fused | 128 | 8 | 0.5671 ms | 605.8 GB/s | 39.0% | 24.1% | 119.2 |
| paged | 32 | 4 | 0.6789 ms | 396.5 GB/s | 25.5% | 27.4% | 72 |
| paged | 64 | 4 | 0.4876 ms | 574.9 GB/s | 37.0% | 26.0% | 96 |
| paged | 128 | 4 | 0.4284 ms | 657.2 GB/s | 42.3% | 15.5% | 168 |
| paged | 128 | 8 | 0.5588 ms | 491.0 GB/s | 31.6% | 22.7% | 128 |

归因结论：

- `block_n=32` 慢的主因不是寄存器，而是访存/调度效率下降。以 fused s2048 为例，最佳 `bn128,nw4` 是 `0.2555 ms / 59.1% DRAM throughput / 18.0% occupancy`；`bn32,nw4` 是 `0.3801 ms / 38.0% / 39.1%`。occupancy 翻高但 latency 慢约 `+48.8%`。
- `num_warps=8` 慢的主因也不是 occupancy 不够，而是每个 program 的有效工作/访存效率下降。fused s2048 最佳 `nw8` 比 `bn128,nw4` 慢约 `+49.1%`，DRAM throughput 从 `59.1%` 降到 `40.2%`。
- paged uniform 相对 fused 的 gap 主要体现为更多 DRAM bytes 和更低 DRAM throughput。s2048 最优点中，fused NCU 窗口约 `855 MB / 918.9 GB/s / 59.1%`，paged 约 `1409 MB / 790.7 GB/s / 50.8%`；s4096 中 fused 约 `1666 MB / 931.6 GB/s / 59.9%`，paged 约 `2761 MB / 789.4 GB/s / 50.8%`。
- paged 的额外成本很可能来自 block_table 间接寻址、physical block 非连续访问、额外地址计算和较差 coalescing/cache behavior。fast metrics 只能证明“多读/低 throughput”，不能定位到具体哪一行。
- paged layout 在 s2048 更敏感：`bn128,nw4,uniform` 下 contiguous/interleaved/shuffled event latency 分别为 `0.2859/0.2891/0.3572 ms`。但这三个点的 fast metrics DRAM throughput 很接近，说明 event latency 差异可能来自 source-line 级 stall、cache/scoreboard 或测量波动；需要 source-line/instruction 报告确认。

下一步真正的 source-line 归因应只跑少数点：

1. fused best: `s2048/d128, bn128, nw4, split4`；
2. fused slow tile: `s2048/d128, bn32, nw4, split4`；
3. fused slow warps: `s2048/d128, bn128, nw8, split1/4`；
4. paged best: `s2048/d128, bn128, nw4, uniform, contiguous`；
5. paged locality slow: `s2048/d128, bn128, nw4, uniform, shuffled`。

重点看 `SourceCounters / InstructionStats / MemoryWorkloadAnalysis / SchedulerStats` 中的 integer/address 指令、global load efficiency、L1/L2 hit、long scoreboard、issue stall 和 local memory spill。

**2026-07-15 source-line / instruction 归因结果（`results/ncu_source_attribution*`）**

本轮已经采集并导出 6 个 `.ncu-rep`：

- fused best: `s2048/d128, bn128, nw4, split4`；
- fused slow tile: `s2048/d128, bn32, nw4, split4`；
- fused slow warps: `s2048/d128, bn128, nw8, split1`；
- fused slow warps: `s2048/d128, bn128, nw8, split4`；
- paged best: `s2048/d128, bn128, nw4, uniform, contiguous`；
- paged locality slow: `s2048/d128, bn128, nw4, uniform, shuffled`。

主 kernel 对比：

| case | NCU kernel time | regs | theo occ | DRAM BW | issue/cycle | active warps | eligible warps | main stalls |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fused `bn128,nw4,split4` | `303.0 us` | 168 | 18.75% | `901.4 GB/s` | 0.48 | 2.86 | 0.72 | wait 24.3%, long scoreboard 13.7%, short scoreboard 12.3%, mio 9.0% |
| fused `bn32,nw4,split4` | `473.8 us` | 72 | 43.75% | `576.9 GB/s` | 0.42 | 6.28 | 0.82 | mio 23.4%, short scoreboard 23.1%, long scoreboard 13.8% |
| fused `bn128,nw8,split1` | `447.0 us` | 120 | 25.00% | `608.2 GB/s` | 0.36 | 3.63 | 0.62 | mio 23.0%, long scoreboard 18.1%, wait 14.0% |
| fused `bn128,nw8,split4` | `481.4 us` | 119 | 25.00% | `567.3 GB/s` | 0.35 | 3.89 | 0.57 | long scoreboard 30.8%, mio 16.7%, wait 13.8% |
| paged contiguous `bn128,nw4` | `353.6 us` | 168 | 18.75% | `768.9 GB/s` | 0.43 | 2.48 | 0.60 | wait 26.7%, long scoreboard 15.8%, short scoreboard 11.6%, mio 8.8% |
| paged shuffled `bn128,nw4` | `355.1 us` | 168 | 18.75% | `765.7 GB/s` | 0.43 | 2.48 | 0.59 | wait 25.8%, long scoreboard 16.8%, short scoreboard 11.4%, mio 9.0% |

归因更新：

- `block_n=32` 的慢因已确认不是寄存器/occupancy。它把寄存器从 168 降到 72、理论 occupancy 从 18.75% 拉到 43.75%，但主 kernel 时间从 `303 us` 增到 `474 us`，DRAM BW 从 `901 GB/s` 掉到 `577 GB/s`，`mio_throttle` 和 `short_scoreboard` 都升到约 23%。这说明小 tile 提高了表面 occupancy，但增加了每个 token 的 shared-memory/调度压力和指令开销，反而降低有效访存吞吐。
- `num_warps=8` 也不是可行方向。`split1` 已经比 best 慢，`split4` 更慢；`split4,nw8` 的 long scoreboard 升到 `30.8%`，eligible warps 只有 `0.57`，issue/cycle 只有 `0.35`。更多 warps 没有隐藏访存延迟，反而扩大 scoreboard 等待和调度低效。
- paged 相对 fused 的 s2048 gap 在 source attribution 中表现为更低 DRAM BW / issue efficiency，而不是 local spill 或 L2 sector inefficiency：paged contiguous 主 kernel `768.9 GB/s / issue 0.43`，fused best 是 `901.4 GB/s / issue 0.48`；两者 L2 theoretical sectors 都等于 ideal，`local_sectors=0`。
- paged contiguous 与 shuffled 在这次 source report 中几乎完全一致：kernel time `353.6 us` vs `355.1 us`，DRAM BW `768.9` vs `765.7 GB/s`，stall 分布也接近。之前 full matrix 中 shuffled 的 event latency 明显慢，更可能是运行波动或外部测量窗口差异；当前没有证据表明 shuffled layout 在这个 s2048/uniform 点引入了稳定的 memory coalescing/cache 问题。
- 具体下一步不应优先继续降寄存器或追 occupancy，而应优先减少 `bn128,nw4` 路径上的无效指令和等待：检查 shared staging / dequant / softmax 更新里的 MIO、short scoreboard 来源；paged 路径重点看 block table 地址计算和额外 global load 是否能合并或搬出内层循环。

**2026-07-15 profiling 结论落地到代码**

- `triton_fused` 默认参数收敛到 `block_n=128, num_warps=4`，长序列默认 `num_splits=4`；短序列仍保留单 kernel 快路径。
- `triton_paged` 默认参数收敛到 `block_n=128, num_warps=4`。
- 保留 `FLASHSPEC_BLOCK_N`、`FLASHSPEC_NUM_WARPS`、`FLASHSPEC_NUM_SPLITS` 覆盖，用于后续 A/B 和回归定位。
- paged kernel 对当前主路径 `page_block_size == quant_block_size` 增加编译期快路径，避免内层循环里恒为 0 的 `physical_quant_block` 除法/乘法地址计算；非等长配置保留通用路径。
- 新增 `scripts/analyze_matrix.py`，从 `results/profile_matrix` 和 `results/ncu_source_attribution_export` 生成稳定 Markdown 报告 `results/profile_matrix_report.md`。

**2026-07-15 文档与脚本收口**

根据 full matrix 结果，项目文档从“设计草案 + 多份历史说明”收口为三份主文档：

- 根目录 `README.md`: 唯一项目入口，合并原 `doc/README.MD` 的设计边界、运行命令和当前 A100 结论；
- `doc/TODO.MD`: 只保留下一阶段任务，不再混入已完成方案；
- `doc/optimization-log.md`: 保留完整实验历史、数据和结论。

删除的过时文档/脚本：

- `doc/README.MD`: 内容已合并进根 README；
- `doc/split-k-plan.md`: Split-K 已实现并通过矩阵 profiling 验证，历史方案由本日志承接；
- `doc/deep_engineering_project.svg`: 早期技术地图，当前 README/TODO 已替代其项目导航作用；
- `benchmarks/ab_split_k.sh`: 单独 Split-K A/B 已被 `scripts/profile_matrix.py` 覆盖；
- `scripts/profile_roofline.py`: 静态 roofline 草图不再是当前 profiling 主流程。

保留的脚本边界：

- `scripts/profile_matrix.py`: 运行 fused/paged profiling 矩阵；
- `scripts/analyze_results.py`: 汇总单点 JSON 并画图；
- `scripts/profile_report.py`: 单个 JSON 的 Markdown report；
- `scripts/backfill_ncu.py`: 保留为手动 `.ncu-rep`/CSV 回填 fallback，虽然常规路径优先使用 `microbench.py --profile-ncu`。

---

## 附:已修复的基础设施 bug

- **ncu kernel 名正则漏匹配 Split-K kernel**(`microbench.py`):原正则 `fused_dequant_attention_kernel`
  匹配不到新增的 `_fused_dequant_attention_split_kernel`(多了 `_split`)和 `_combine_splits_kernel`。
  后果:Split-K 开启时 `--profile-ncu` 抓不到 kernel,占用率/带宽数据全空。
  修复:正则改为 `fused_dequant_attention|combine_splits`(paged 同步加 `|combine_splits`);
  `ncu_parse.py` 把 `combine_splits` 加进 good_markers 防误报。
