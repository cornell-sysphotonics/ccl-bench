#!/bin/bash
# Bandwidth sweep: how does simulated step time and comm fraction change as we
# scale inter-node network bandwidth from 50 GB/s to 800 GB/s?
#
# Trace: deepseek-v3-16b ep4-dp2-tp4 (8 ranks, no PP).
# Intra-node fabric is held constant at 400 GB/s, 50 ns.

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter
GPUS_PER_NODE=4
INTRA_BW=400
INTRA_LAT=50

echo "=== Scale-out Bandwidth Sweep: deepseek-v3-16b ep4-dp2-tp4 ==="
echo "Fixed scale-up: FullyConnected, ${INTRA_BW} GB/s, ${INTRA_LAT} ns"
echo

for BW in 50 100 200 400 800; do
    OUTDIR=/var/tmp/ccl_bench_sim_bw${BW}

    echo "--- ${BW} GB/s ---"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology FullyConnected \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology Switch \
        --bandwidth "$BW" \
        --latency 500 \
        --collective-algo ring \
        2>&1 | grep -E "Simulated step|comm fraction|ERROR"
    echo
done
