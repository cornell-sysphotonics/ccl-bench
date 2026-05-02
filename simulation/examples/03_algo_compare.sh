#!/bin/bash
# Collective algorithm comparison: ring vs halving_doubling vs doubleBinaryTree.
#
# At moderate scale (8 ranks), ring is typically optimal for bandwidth-bound
# collectives. halving_doubling (recursive halving) reduces latency at the cost
# of more bandwidth. doubleBinaryTree reduces bandwidth for allreduce at scale.
#
# Trace: deepseek-v3-16b ep4-dp2-tp4 (8 ranks, no PP).

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter
GPUS_PER_NODE=4
INTRA_BW=400
INTRA_LAT=50
INTER_BW=200
INTER_LAT=500

echo "=== Collective Algorithm Comparison: deepseek-v3-16b ep4-dp2-tp4 ==="
echo "Topology: FullyConnected scale-up + Switch scale-out"
echo "BW: ${INTRA_BW}/${INTER_BW} GB/s, Latency: ${INTRA_LAT}/${INTER_LAT} ns"
echo

for ALGO in ring halving_doubling doubleBinaryTree; do
    OUTDIR=/var/tmp/ccl_bench_sim_algo_${ALGO}

    echo "--- $ALGO ---"
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
        --collective-algo "$ALGO" \
        2>&1 | grep -E "Simulated step|comm fraction|ERROR"
    echo
done
