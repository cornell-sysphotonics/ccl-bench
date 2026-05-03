#!/bin/bash
# Scale-up bandwidth sweep with a 32-GPU scale-up domain.
#
# Varies --intra-bandwidth while holding scale-up domain size and scale-out fixed:
#   scale-up domain: 32 GPUs
#   scale-up topology: FullyConnected, 50 ns
#   scale-out: Switch, 50 GB/s, 50000 ns

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter
# TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep8-dp2-tp4-perlmutter

GPUS_PER_NODE=32
# GPUS_PER_NODE=8
INTRA_TOPOLOGY=FullyConnected
INTRA_LAT=50
INTER_TOPOLOGY=Switch
INTER_BW=50
INTER_LAT=50000
COLLECTIVE_ALGO=ring
# COMPUTE_MODEL=gap
COMPUTE_MODEL=kernels

SCALEUP_BANDWIDTHS=(112.5 225 450 900 1800 3600)

echo "=== Scale-up Bandwidth Sweep: deepseek-v3-16b ep32, scale-up domain size ${GPUS_PER_NODE} ==="
echo "Trace: $TRACE"
echo "Scale-up topology: ${INTRA_TOPOLOGY}, Latency: ${INTRA_LAT} ns"
echo "Scale-out topology: ${INTER_TOPOLOGY}, BW: ${INTER_BW} GB/s, Latency: ${INTER_LAT} ns"
echo "Collective algo: ${COLLECTIVE_ALGO}"
echo "Compute model: ${COMPUTE_MODEL}"
echo

SUMMARY="$REPO/simulation/examples/scaleup_bw_sweep_kernels/scaleup_bandwidth_sweep_summary.tsv"
mkdir -p "$(dirname "$SUMMARY")"
printf "intra_bandwidth_GBps\tstep_ms\tcomm_fraction_pct\n" > "$SUMMARY"

# PREV_OUTDIR="bw_sweep/bw_sweep_ep32_5"
PREV_OUTDIR=""
for INTRA_BW in "${SCALEUP_BANDWIDTHS[@]}"; do
    OUTDIR="$REPO/simulation/examples/scaleup_bw_sweep_$COMPUTE_MODEL/su_ep32_${GPUS_PER_NODE}_bw${INTRA_BW}_${INTRA_TOPOLOGY}_inter${INTER_TOPOLOGY}"
    LOG="$OUTDIR/sweep_stdout.log"
    REUSE_ARGS=()
    if [ -n "$PREV_OUTDIR" ]; then
        REUSE_ARGS=(--reuse-et-from "$PREV_OUTDIR")
    fi

    echo "--- scale-up bandwidth ${INTRA_BW} GB/s ---"
    mkdir -p "$OUTDIR"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        "${REUSE_ARGS[@]}" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology "$INTRA_TOPOLOGY" \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology "$INTER_TOPOLOGY" \
        --bandwidth "$INTER_BW" \
        --latency "$INTER_LAT" \
        --collective-algo "$COLLECTIVE_ALGO" \
        --compute-model "$COMPUTE_MODEL" \
        --kernel-dependency-mode rank \
        2>&1 | tee "$LOG"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET|Hardware:" "$LOG" || true

    RESULT_LINE=$(grep -E "Simulated step time:" "$LOG" | tail -n 1 || true)
    if [ -n "$RESULT_LINE" ]; then
        STEP_MS=$(awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$RESULT_LINE")
        COMM_PCT=$(awk -F'Comm fraction: ' '{print $2}' <<< "$RESULT_LINE" | awk '{gsub("%", "", $1); print $1}')
        printf "%s\t%s\t%s\n" "$INTRA_BW" "$STEP_MS" "$COMM_PCT" >> "$SUMMARY"
    fi

    PREV_OUTDIR="$OUTDIR"
    echo
done
