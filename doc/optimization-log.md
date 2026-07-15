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

## 附:已修复的基础设施 bug

- **ncu kernel 名正则漏匹配 Split-K kernel**(`microbench.py`):原正则 `fused_dequant_attention_kernel`
  匹配不到新增的 `_fused_dequant_attention_split_kernel`(多了 `_split`)和 `_combine_splits_kernel`。
  后果:Split-K 开启时 `--profile-ncu` 抓不到 kernel,占用率/带宽数据全空。
  修复:正则改为 `fused_dequant_attention|combine_splits`(paged 同步加 `|combine_splits`);
  `ncu_parse.py` 把 `combine_splits` 加进 good_markers 防误报。
