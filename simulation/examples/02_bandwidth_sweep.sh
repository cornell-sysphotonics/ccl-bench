#!/bin/bash
# Bandwidth sweep: how does simulated step time and comm fraction change as we
# scale inter-node network bandwidth from 50 GB/s to 800 GB/s?
#
# Trace: deepseek-v3-16b ep4-dp2-tp4 (8 ranks, no PP).
# Intra-node fabric is held constant at 400 GB/s, 50 ns.

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
# TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep8-dp2-tp4-perlmutter
# TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter
GPUS_PER_NODE=4
INTRA_BW=300
INTRA_LAT=25

echo "=== Scale-out Bandwidth Sweep: deepseek-v3-16b ep4-dp2-tp4 ==="
echo "Fixed scale-up: FullyConnected, ${INTRA_BW} GB/s, ${INTRA_LAT} ns"
echo

PREV_OUTDIR=""
for BW in 1.25 5 12.5 25 50 100 200; do
    # OUTDIR="$REPO/simulation/examples/ccl_bench_sim_bw${BW}_ep8"
    # OUTDIR="$REPO/simulation/examples/ccl_bench_sim_bw${BW}_ep4_kernels"
    OUTDIR="$REPO/simulation/examples/bw_sweep/bw_sweep_${BW}"
    REUSE_ARGS=()
    if [ -n "$PREV_OUTDIR" ]; then
        REUSE_ARGS=(--reuse-et-from "$PREV_OUTDIR")
    fi

    echo "--- ${BW} GB/s ---"
    LOG="$OUTDIR/sweep_stdout.log"
    mkdir -p "$OUTDIR"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        "${REUSE_ARGS[@]}" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology Ring \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology Switch \
        --bandwidth "$BW" \
        --latency 50000 \
        --collective-algo ring \
        --compute-model gap \
        2>&1 | tee "$LOG"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET" "$LOG" || true
    PREV_OUTDIR="$OUTDIR"
    echo
done
