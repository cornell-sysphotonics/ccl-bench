#!/bin/bash
# Hardware generation comparison: A100 vs H100 vs hypothetical next-gen.
#
# In comm-only mode the compute time is replayed from measurements (A100 actuals),
# so only the network parameters vary. Pair with --mode full and real H100 traces
# for a fair apples-to-apples comparison.
#
# A100-Perlmutter: 400 GB/s NVLink-class scale-up + 200 GB/s Slingshot scale-out
# H100-IB:         900 GB/s NVLink scale-up + 400 GB/s InfiniBand scale-out
# H100-NVLink:     Faster scale-up only; scale-out held at A100 baseline
# Next-gen:        Speculative faster scale-up and scale-out
#
# Trace: deepseek-v3-16b ep4-dp2-tp4 (8 ranks, no PP).

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter
GPUS_PER_NODE=4

echo "=== Hardware Generation Comparison: deepseek-v3-16b ep4-dp2-tp4 ==="
echo "(Compute time from A100 measurements; only network simulated)"
echo

declare -A HW_INTRA_BW=(
    ["A100-Perlmutter"]=400
    ["H100-IB"]=900
    ["H100-NVLink-only"]=900
    ["next-gen"]=1600
)
declare -A HW_INTER_BW=(
    ["A100-Perlmutter"]=200
    ["H100-IB"]=400
    ["H100-NVLink-only"]=200
    ["next-gen"]=800
)
declare -A HW_INTRA_LAT=(
    ["A100-Perlmutter"]=50
    ["H100-IB"]=30
    ["H100-NVLink-only"]=30
    ["next-gen"]=20
)
declare -A HW_INTER_LAT=(
    ["A100-Perlmutter"]=500
    ["H100-IB"]=300
    ["H100-NVLink-only"]=500
    ["next-gen"]=150
)

for HW in A100-Perlmutter H100-IB H100-NVLink-only next-gen; do
    INTRA_BW=${HW_INTRA_BW[$HW]}
    INTER_BW=${HW_INTER_BW[$HW]}
    INTRA_LAT=${HW_INTRA_LAT[$HW]}
    INTER_LAT=${HW_INTER_LAT[$HW]}

    OUTDIR=/var/tmp/ccl_bench_sim_hw_${HW}

    echo "--- $HW (scale-up ${INTRA_BW} GB/s ${INTRA_LAT} ns, scale-out ${INTER_BW} GB/s ${INTER_LAT} ns) ---"
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
