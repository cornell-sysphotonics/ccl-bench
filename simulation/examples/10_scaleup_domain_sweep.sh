#!/bin/bash
# Scale-up domain size sweep for the ep32 DeepSeek trace.
#
# Varies --gpus-per-node while holding bandwidths fixed:
#   scale-up:  450 GB/s, 50 ns
#   scale-out: 50 GB/s, 50000 ns

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter

INTRA_TOPOLOGY=FullyConnected
INTER_TOPOLOGY=Switch
INTRA_BW=450
INTRA_LAT=50
INTER_BW=50
INTER_LAT=50000
COLLECTIVE_ALGO=ring
COMPUTE_MODEL=gap

SCALEUP_DOMAIN_SIZES=(4 8 16 32)

echo "=== Scale-up Domain Size Sweep: deepseek-v3-16b ep32 ==="
echo "Trace: $TRACE"
echo "Scale-up topology: ${INTRA_TOPOLOGY}, BW: ${INTRA_BW} GB/s, Latency: ${INTRA_LAT} ns"
echo "Scale-out topology: ${INTER_TOPOLOGY}, BW: ${INTER_BW} GB/s, Latency: ${INTER_LAT} ns"
echo "Collective algo: ${COLLECTIVE_ALGO}"
echo "Compute model: ${COMPUTE_MODEL}"
echo

SUMMARY="$REPO/simulation/examples/scaleup_domain_sweep/scaleup_domain_sweep_summary.tsv"
mkdir -p "$(dirname "$SUMMARY")"
printf "gpus_per_node\tstep_ms\tcomm_fraction_pct\n" > "$SUMMARY"

PREV_OUTDIR="bw_sweep/bw_sweep_ep32_5"
for GPUS_PER_NODE in "${SCALEUP_DOMAIN_SIZES[@]}"; do
    OUTDIR="$REPO/simulation/examples/scaleup_domain_sweep/su${GPUS_PER_NODE}_${INTRA_TOPOLOGY}_inter${INTER_TOPOLOGY}"
    LOG="$OUTDIR/sweep_stdout.log"
    REUSE_ARGS=()
    if [ -n "$PREV_OUTDIR" ]; then
        REUSE_ARGS=(--reuse-et-from "$PREV_OUTDIR")
    fi

    echo "--- scale-up domain size ${GPUS_PER_NODE} ---"
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
        2>&1 | tee "$LOG"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET|Hardware:" "$LOG" || true

    RESULT_LINE=$(grep -E "Simulated step time:" "$LOG" | tail -n 1 || true)
    if [ -n "$RESULT_LINE" ]; then
        STEP_MS=$(awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$RESULT_LINE")
        COMM_PCT=$(awk -F'Comm fraction: ' '{print $2}' <<< "$RESULT_LINE" | awk '{gsub("%", "", $1); print $1}')
        printf "%s\t%s\t%s\n" "$GPUS_PER_NODE" "$STEP_MS" "$COMM_PCT" >> "$SUMMARY"
    fi
    
    PREV_OUTDIR="$OUTDIR"
    echo
done
