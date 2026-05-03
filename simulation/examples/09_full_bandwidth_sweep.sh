#!/bin/bash
# Full-mode scale-out bandwidth sweep for llama-3.1-8b Torchtitan traces.
#
# Keeps H100-like compute and scale-up parameters fixed:
#   peak perf: 900 TFLOPS
#   HBM BW:    3350 GB/s
#   scale-up:  8 GPUs/node, 900 GB/s
# Sweeps scale-out bandwidth around the 50 GB/s baseline.

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE="$REPO/trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter"

PEAK_PERF=900
MEM_BW=3350
GPUS_PER_NODE=4
INTRA_BW=300
INTRA_LAT=50
INTER_LAT=500

BANDWIDTHS=(5 12.5 25 50 100 200)

echo "=== Full-mode Scale-out Bandwidth Sweep: llama-3.1-8b ==="
echo "Trace: $TRACE"
echo "Compute: peak=${PEAK_PERF} TFLOPS, mem-bw=${MEM_BW} GB/s"
echo "Scale-up: ${GPUS_PER_NODE} GPUs/node, ${INTRA_BW} GB/s, ${INTRA_LAT} ns"
echo "Scale-out latency: ${INTER_LAT} ns"
echo

PREV_OUTDIR="${REUSE_ET_FROM:-}"
for BW in "${BANDWIDTHS[@]}"; do
    OUTDIR="$REPO/simulation/examples/llama_full_bw_sweep/llama31_8b_bw${BW}"
    LOG="$OUTDIR/sweep_stdout.log"
    REUSE_ARGS=()
    if [ -n "$PREV_OUTDIR" ]; then
        REUSE_ARGS=(--reuse-et-from "$PREV_OUTDIR")
    fi

    echo "--- scale-out bandwidth ${BW} GB/s ---"
    mkdir -p "$OUTDIR"
    python "$REPO/simulation/pipeline.py" \
        --mode full \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        "${REUSE_ARGS[@]}" \
        --peak-perf "$PEAK_PERF" \
        --mem-bw "$MEM_BW" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology FullyConnected \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology Switch \
        --bandwidth "$BW" \
        --latency "$INTER_LAT" \
        --collective-algo ring \
        2>&1 | tee "$LOG"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Mode:|Hardware:" "$LOG" || true
    PREV_OUTDIR="$OUTDIR"
    echo
done
