#!/bin/bash
# Parallelism strategy comparison: same model + hardware, different EP/TP/DP split.
#
# All configs use 8 GPUs on A100+Slingshot with two-tier networking. The traces
# capture measured compute time, so differences in simulated step time reflect
# real compute + simulated communication under the given network parameters.
#
#   ep4-dp2-tp4  : EP=4, DP=2, TP=4  (balanced)
#   ep4-dp4-tp2  : EP=4, DP=4, TP=2  (more data parallelism, less TP bandwidth)
#   ep8-dp2-tp4  : EP=8, DP=2, TP=4  (more expert parallelism)
#   ep8-dp8      : EP=8, DP=8, TP=1  (pure EP+DP, no tensor parallelism)

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BASE=/data/ccl-bench_trace_collection
GPUS_PER_NODE=4
INTRA_BW=400
INTRA_LAT=50
INTER_BW=200
INTER_LAT=500

echo "=== Parallelism Strategy Comparison: deepseek-v3-16b (8 GPUs) ==="
echo "Hardware: FullyConnected scale-up + Switch scale-out, ring"
echo "BW: ${INTRA_BW}/${INTER_BW} GB/s, Latency: ${INTRA_LAT}/${INTER_LAT} ns"
echo

declare -A CONFIGS=(
    ["ep4-dp2-tp4"]="deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter"
    ["ep4-dp4-tp2"]="deepseek-v3-16b-torchtitan-ep4-dp4-tp2-perlmutter"
    ["ep8-dp2-tp4"]="deepseek-v3-16b-torchtitan-ep8-dp2-tp4-perlmutter"
    ["ep8-dp8"]="deepseek-v3-16b-torchtitan-ep8-dp8-perlmutter"
)

for LABEL in ep4-dp2-tp4 ep4-dp4-tp2 ep8-dp2-tp4 ep8-dp8; do
    TRACE="$BASE/${CONFIGS[$LABEL]}"
    OUTDIR=/var/tmp/ccl_bench_sim_${LABEL}

    echo "--- $LABEL ---"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology FullyConnected \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology Switch \
        --bandwidth "$INTER_BW" \
        --latency "$INTER_LAT" \
        --collective-algo ring \
        2>&1 | grep -E "Simulated step|comm fraction|ERROR"
    echo
done
