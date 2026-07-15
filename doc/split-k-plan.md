# Split-K (Flash-Decoding) 改造方案

## 目标
把 decode attention 的并行任务数从 `batch*heads`(=512)提升到 `batch*heads*S`,
填满 A100 的 108 个 SM,解决 occupancy ~25% 的根因。预期长序列(2048/4096)显著提速。

## 背景 / 为什么
- 现状:每个 program 管一个 [batch,head],串行扫完整 seq_len。grid=512,单 wave 发完,
  每 SM 只落 ~4.7 个 program,占用率钉死在 ~25%。
- 已否掉的便宜路:num_warps 4→8 全面变慢 27–78%(每 program 活太小,跨 warp 汇总开销翻倍)。
- 正路:Split-K —— 把"扫 seq_len"沿序列维切成 S 段并行,每段活量不变,只增加任务数。

## 核心设计:两段式(split kernel + combine kernel)
1. **Split kernel**:grid = (batch*heads, S)。每个 program 只扫自己那段
   `[split*chunk, (split+1)*chunk)` 的 token,产出该段的 online-softmax 部分状态:
   - partial_m [batch,heads,S]       段内最大值
   - partial_l [batch,heads,S]       段内 exp 累积和
   - partial_acc [batch,heads,S,head_dim]  段内未归一化 PV 累积
   写入 scratch 显存(不是最终 out)。
2. **Combine kernel**:grid = (batch*heads,)。每个 program 读自己 S 段的 partial 状态,
   做标准 online-softmax 合并:
   - m* = max_s(partial_m[s])
   - out = Σ_s exp(partial_m[s]-m*) * partial_acc[s]  /  Σ_s exp(partial_m[s]-m*) * partial_l[s]
   写入最终 out [batch,heads,head_dim]。

合并数学与现有 kernel 内的 online-softmax rescale 完全同构,只是跨"段"而非跨"block"。

## num_splits (S) 如何取
自适应:`S = clamp(ceil(seq_len / TOKENS_PER_SPLIT), 1, S_MAX)`,建议 TOKENS_PER_SPLIT=512, S_MAX=32。
- s512  → S=1(退化回单 kernel 路径,零 combine 开销)
- s2048 → S=4
- s4096 → S=8
每段 chunk = ceil(seq_len / S),尾段用 mask 处理不整除。
S==1 时走快路径:直接用现有单 kernel 写 out,不分配 scratch、不启 combine。

## 改动范围(文件级)
- `src/flashspec/triton_fused.py`
  - 现 kernel 增加 `split_id`/`num_splits`/`chunk` 维度与写 partial 的逻辑
    (或新增 `_fused_dequant_attention_split_kernel`,保留原 kernel 作 S==1 快路径)。
  - 新增 `_combine_splits_kernel`(fused/paged 可共用,放 triton_utils 或各自文件)。
  - launcher `_run_fused_dequant_attention_triton`:算 S、分配 scratch、启两个 kernel。
- `src/flashspec/triton_paged.py`:同样改造(paged 的间接寻址逻辑不变,只加 split 分段 + 复用 combine)。
- `num_warps`:回退到 4 作为 split kernel 起点(后续可 autotune)。
- 无需改 API:`fused_dequant_attention_triton` / `paged_quant_attention_triton` 签名与返回不变,
  stats 里 `materializes_dense_kv` 仍为 0.0(scratch 是 partial 状态,非 dense KV;字节量远小)。

## scratch 显存开销(可接受)
s4096/d128, S=8: partial_acc = 16*32*8*128*4B ≈ 16.8MB,partial_m/l 可忽略。用 torch.empty 分配。

## 正确性验证(关键,GPU 上做)
- 复用现有 tests/test_flashspec.py 的 CUDA 用例(rtol/atol=2e-2 对齐反量化参考)。
  合并数学正确的话,Split-K 输出应与单 kernel 数值一致 —— 现有测试足够覆盖,
  但需额外加一个"强制 S>1(长 seq)"的用例,确保 combine 路径被测到。
- 单 split(S==1)与多 split(S>1)都要过 correctness。

## Benchmark 验证(Colab A100)
重跑 --profile-ncu,对比:
- measured_occupancy_pct: ~25% → 期望显著上升(长序列最明显)
- latency_ms: 长序列(2048/4096)期望下降;短序列(512,S=1)应与现状持平(未退化)
- 判读:若 occupancy 升且长序列延迟降 → Split-K 生效;若短序列退化 → 检查 S==1 快路径是否真的零开销。

## 风险 / 诚实说明
- 本机无 GPU,所有 Triton 改动只能保证"能 import/语法对",数值正确性与性能必须在 Colab 验证。
- Split-K 对短序列可能因 combine 固定开销略有损耗,靠 S==1 快路径规避。
- 合并数学若 rescale 写错会静默产生错误结果 —— 依赖 correctness 测试兜底,先加强测试再看性能。
- 首版目标是"正确 + 长序列变快",kernel 微调(block_n / num_warps / autotune)留到方向验证后。

## 实施顺序
1. 回退 num_warps 8→4(两个 kernel)。
2. 先做 fused:加 split kernel + combine kernel + launcher 自适应 S + S==1 快路径。
3. 加强 correctness 测试(强制 S>1 的长 seq 用例)。
4. fused 在 Colab 验证正确性 + occupancy/latency,方向确认后再复制到 paged。
5. 全部 push,Colab 重跑完整 sweep,更新 analyze 图。
6. (可选)autotune num_warps/block_n/S。
```
```
