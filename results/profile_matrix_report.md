# FlashSpec Profiling Matrix Report

- matrix_dir: `results\profile_matrix`
- source_dir: `results\ncu_source_attribution_export`
- matrix_points: `120`

## Top Latency Points

### triton_fused / 2048 / 128

| backend | seq | head_dim | latency ms | block_n | warps | splits | len pattern | layout | BW GB/s | DRAM % | occ % | regs |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| triton_fused | 2048 | 128 | 0.2555 | 128.0 | 4.0 | 4.0 | uniform | contiguous | 918.9 | 59.1 | 18.0 | 168 |
| triton_fused | 2048 | 128 | 0.2652 | 128.0 | 4.0 | 8.0 | uniform | contiguous | 909.7 | 58.5 | 18.3 | 168 |
| triton_fused | 2048 | 128 | 0.2743 | 128.0 | 4.0 | 1.0 | uniform | contiguous | 824.7 | 53.0 | 15.5 | 168 |
| triton_fused | 2048 | 128 | 0.2978 | 64.0 | 4.0 | 8.0 | uniform | contiguous | 786.3 | 50.6 | 24.1 | 114 |
| triton_fused | 2048 | 128 | 0.3066 | 64.0 | 4.0 | 1.0 | uniform | contiguous | 740.3 | 47.6 | 24.9 | 96 |

### triton_fused / 4096 / 128

| backend | seq | head_dim | latency ms | block_n | warps | splits | len pattern | layout | BW GB/s | DRAM % | occ % | regs |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| triton_fused | 4096 | 128 | 0.4856 | 128.0 | 4.0 | 4.0 | uniform | contiguous | 931.6 | 59.9 | 18.0 | 168 |
| triton_fused | 4096 | 128 | 0.4883 | 128.0 | 4.0 | 8.0 | uniform | contiguous | 938.0 | 60.3 | 18.3 | 168 |
| triton_fused | 4096 | 128 | 0.4883 | 128.0 | 4.0 | 8.0 | uniform | contiguous | 937.6 | 60.3 | 18.3 | 168 |
| triton_fused | 4096 | 128 | 0.5365 | 128.0 | 4.0 | 1.0 | uniform | contiguous | 825.8 | 53.1 | 15.5 | 168 |
| triton_fused | 4096 | 128 | 0.5708 | 64.0 | 4.0 | 8.0 | uniform | contiguous | 789.6 | 50.8 | 24.2 | 114 |

### triton_paged / 2048 / 128

| backend | seq | head_dim | latency ms | block_n | warps | splits | len pattern | layout | BW GB/s | DRAM % | occ % | regs |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| triton_paged | 2048 | 128 | 0.2770 | 128.0 | 4.0 |  | descending | contiguous | 527.7 | 33.9 | 15.6 | 168 |
| triton_paged | 2048 | 128 | 0.2858 | 128.0 | 4.0 |  | uniform | contiguous | 790.7 | 50.8 | 15.5 | 168 |
| triton_paged | 2048 | 128 | 0.2891 | 128.0 | 4.0 |  | uniform | interleaved | 786.1 | 50.6 | 15.5 | 168 |
| triton_paged | 2048 | 128 | 0.3024 | 64.0 | 4.0 |  | descending | contiguous | 473.1 | 30.4 | 26.2 | 96 |
| triton_paged | 2048 | 128 | 0.3085 | 128.0 | 4.0 |  | descending | interleaved | 527.1 | 33.9 | 15.6 | 168 |

### triton_paged / 4096 / 128

| backend | seq | head_dim | latency ms | block_n | warps | splits | len pattern | layout | BW GB/s | DRAM % | occ % | regs |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| triton_paged | 4096 | 128 | 0.5289 | 128.0 | 4.0 |  | descending | interleaved | 524.0 | 33.7 | 15.6 | 168 |
| triton_paged | 4096 | 128 | 0.5298 | 128.0 | 4.0 |  | descending | contiguous | 522.5 | 33.6 | 15.6 | 168 |
| triton_paged | 4096 | 128 | 0.5319 | 128.0 | 4.0 |  | descending | shuffled | 521.0 | 33.5 | 15.6 | 168 |
| triton_paged | 4096 | 128 | 0.5604 | 128.0 | 4.0 |  | uniform | contiguous | 789.4 | 50.8 | 15.5 | 168 |
| triton_paged | 4096 | 128 | 0.5622 | 128.0 | 4.0 |  | uniform | interleaved | 799.6 | 51.4 | 15.5 | 168 |

## Parameter Averages


### triton_fused: block_n / num_warps

| block_n | num_warps | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---|---|---|---|---|---|---|---|
| 128.0 | 4.0 | 8 | 0.3880 | 900.7 | 57.9 | 17.5 | 168 |
| 128.0 | 8.0 | 8 | 0.5671 | 605.8 | 39.0 | 24.1 | 119 |
| 32.0 | 4.0 | 8 | 0.5910 | 567.6 | 36.5 | 36.7 | 74 |
| 32.0 | 8.0 | 8 | 1.1275 | 294.7 | 19.0 | 35.2 | 80 |
| 64.0 | 4.0 | 8 | 0.4478 | 771.8 | 49.6 | 24.1 | 110 |
| 64.0 | 8.0 | 8 | 0.7486 | 448.2 | 28.8 | 26.3 | 106 |

### triton_fused: num_splits

| num_splits | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---|---|---|---|---|---|---|
| 1.0 | 12 | 0.6690 | 566.3 | 36.4 | 25.3 | 104 |
| 4.0 | 18 | 0.5733 | 605.2 | 38.9 | 27.6 | 111 |
| 8.0 | 18 | 0.7007 | 612.3 | 39.4 | 28.3 | 111 |

### triton_paged: block_n / num_warps

| block_n | num_warps | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---|---|---|---|---|---|---|---|
| 128.0 | 4.0 | 12 | 0.4284 | 657.2 | 42.3 | 15.5 | 168 |
| 128.0 | 8.0 | 12 | 0.5588 | 491.0 | 31.6 | 22.7 | 128 |
| 32.0 | 4.0 | 12 | 0.6789 | 396.5 | 25.5 | 27.4 | 72 |
| 32.0 | 8.0 | 12 | 1.2063 | 223.1 | 14.3 | 31.6 | 80 |
| 64.0 | 4.0 | 12 | 0.4876 | 574.9 | 37.0 | 26.0 | 96 |
| 64.0 | 8.0 | 12 | 0.8391 | 322.0 | 20.7 | 22.8 | 116 |

### triton_paged: paged_layout

| paged_layout | points | avg latency ms | avg BW GB/s | avg DRAM % | avg occ % | avg regs |
|---|---|---|---|---|---|---|
| contiguous | 24 | 0.6935 | 443.7 | 28.5 | 24.3 | 110 |
| interleaved | 24 | 0.6966 | 446.0 | 28.7 | 24.3 | 110 |
| shuffled | 24 | 0.7095 | 442.7 | 28.5 | 24.3 | 110 |

## Source Attribution Summary

| case | time us | regs | theo occ % | DRAM GB/s | issue/cycle | eligible warps | long % | short % | wait % | mio % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fused_best_s2048_d128_bn128_nw4_split4 | 303.0 | 168 | 18.75 | 901.4 | 0.48 | 0.72 | 13.7 | 12.3 | 24.3 | 9.0 |
| fused_slow_tile_s2048_d128_bn32_nw4_split4 | 473.8 | 72 | 43.75 | 576.9 | 0.42 | 0.82 | 13.8 | 23.1 | 10.0 | 23.4 |
| fused_slow_warps_s2048_d128_bn128_nw8_split1 | 447.0 | 120 | 25.00 | 608.2 | 0.36 | 0.62 | 18.1 | 12.7 | 14.0 | 23.0 |
| fused_slow_warps_s2048_d128_bn128_nw8_split4 | 481.4 | 119 | 25.00 | 567.3 | 0.35 | 0.57 | 30.8 | 11.2 | 13.8 | 16.7 |
| paged_best_s2048_d128_bn128_nw4_uniform_contiguous | 353.6 | 168 | 18.75 | 768.9 | 0.43 | 0.60 | 15.8 | 11.6 | 26.7 | 8.8 |
| paged_locality_slow_s2048_d128_bn128_nw4_uniform_shuffled | 355.1 | 168 | 18.75 | 765.7 | 0.43 | 0.59 | 16.8 | 11.4 | 25.8 | 9.0 |
