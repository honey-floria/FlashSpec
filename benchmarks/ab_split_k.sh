#!/usr/bin/env bash
# Split-K A/B：固定 num_warps=4，只切换 Split-K 开/关，隔离其纯贡献。
# 每个 seq_len 跑两次：OFF(FLASHSPEC_NUM_SPLITS=1 走单 kernel) 和 ON(自适应 S)。
# 用法：bash benchmarks/ab_split_k.sh
#      SEQ_LENS="2048 4096" HEAD_DIM=128 bash benchmarks/ab_split_k.sh
set -euo pipefail

BATCH=${BATCH:-16}
HEADS=${HEADS:-32}
HEAD_DIM=${HEAD_DIM:-128}
BLOCK_SIZE=${BLOCK_SIZE:-16}
SEQ_LENS=${SEQ_LENS:-"2048 4096"}
OUTDIR=${OUTDIR:-results}
mkdir -p "$OUTDIR"

common="--backend triton_fused --batch $BATCH --heads $HEADS --head-dim $HEAD_DIM --block-size $BLOCK_SIZE --iters 50 --warmup 10 --repeats 20 --device cuda --dtype float16 --json"

for seq in $SEQ_LENS; do
  echo ">>> seq=$seq  Split-K OFF (单 kernel)"
  FLASHSPEC_NUM_SPLITS=1 python benchmarks/microbench.py $common \
    --seq-len "$seq" --output "$OUTDIR/ab_s${seq}_splitoff.json"

  echo ">>> seq=$seq  Split-K ON (自适应)"
  python benchmarks/microbench.py $common \
    --seq-len "$seq" --output "$OUTDIR/ab_s${seq}_spliton.json"
done

echo
echo "=== A/B 对比 (num_warps=4 固定) ==="
python3 - "$OUTDIR" $SEQ_LENS <<'PY'
import json, sys
outdir, *seqs = sys.argv[1:]
def lat(d): return d.get("measured_kernel_latency_ms") or d["latency_ms"]
print(f"{'seq':>6} {'OFF ms':>9} {'ON ms':>9} {'ON splits':>10} {'speedup':>8}")
for seq in seqs:
    off = json.load(open(f"{outdir}/ab_s{seq}_splitoff.json"))
    on  = json.load(open(f"{outdir}/ab_s{seq}_spliton.json"))
    lo, ln = lat(off), lat(on)
    sp = lo / ln if ln else float("nan")
    verdict = "Split-K 更快" if sp > 1.03 else ("持平" if sp > 0.97 else "Split-K 更慢")
    print(f"{seq:>6} {lo:>9.4f} {ln:>9.4f} {str(on.get('num_splits')):>10} {sp:>6.2f}x  {verdict}")
PY
