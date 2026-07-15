# FlashSpec Profiling Report

- backend: `triton_fused`
- device: `NVIDIA A100-SXM4-40GB`
- shape: batch=16, heads=32, seq_len=2048, head_dim=128
- timing_method: `cuda_event`
- latency_ms: `0.370156`
- latency_p50_ms: `0.370156`
- latency_p90_ms: `0.370854`
- latency_p99_ms: `0.370988`
- tokens_per_second: `43225.1`
- materializes_dense_kv: `False`
- raw_latency_samples: `20`

## Nsight Compute Command

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed --kernel-name regex:"fused_dequant_attention|combine_splits" --launch-count 5 --target-processes all --export results/ncu_triton_fused_b16_h32_s2048_d128 --force-overwrite --csv python benchmarks/microbench.py --backend triton_fused --batch 16 --heads 32 --seq-len 2048 --head-dim 128 --block-size 16 --iters 50 --warmup 10 --repeats 1 --device cuda --dtype float16 --json
```

## Latency Breakdown Map

| stage | estimated bytes | estimated flops/ops | measurement | notes |
|---|---:|---:|---|---|
| kv_load | 2.69222e+08 | 0 | ncu dram/source-line metrics | 读取 K/V；Triton quant backend 应主要读取 INT8 values 和 scale/zero_point。 |
| dequant | 0 | 5.36871e+08 | ncu source-line/instruction metrics | INT8 -> float 的 zero_point/scale 反量化；portable quant backend 会额外写出 dense KV，Triton backend 不物化 dense KV。 |
| qk | 0 | 2.68435e+08 | ncu source-line/instruction metrics | QK 点积，输出每个历史 token 的 score。 |
| softmax | 0 | 3.14573e+06 | ncu source-line/instruction metrics | online softmax 的 max、exp、sum 更新。 |
| pv_accumulation | 0 | 2.68435e+08 | ncu source-line/instruction metrics | softmax 权重乘 V 并累积输出。 |
| output_write | 262144 | 0 | ncu dram/source-line metrics | 写出 [batch, heads, head_dim] attention 结果。 |
| total_kernel_or_backend | 2.69484e+08 | 1.07689e+09 | cuda_event | 总耗时来自 benchmark 测量；阶段 latency 需要 Nsight Compute 进一步归因。 |

## Measured Profiler Fields

- measured_kernel_latency_ms: `0.370156`
- measured_dram_bytes: `8.55102e+08`
- measured_achieved_bandwidth_gbps: `775.403`
- measured_occupancy_pct: `23.475`
- measured_sm_utilization_pct: `46.3203`

这些 measured profiler 字段需要用上面的 Nsight Compute 命令采集后回填；microbench 默认只负责生成可复现命令和 CUDA event latency。
