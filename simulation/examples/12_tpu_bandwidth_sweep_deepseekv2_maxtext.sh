#!/bin/bash
# TPU ICI bandwidth scaling example for DeepSeek-V2 16B MaxText trace.
#
# Uses simulation/pipeline.py --mode tpu-xla. The pipeline selects one
# jit_train_step iteration, converts XLA HLO/device events to Chakra ET, and
# runs AstraSim on an 8-chip TPU torus modeled as [ Ring, Ring ].

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE_DIR=/data/ccl-bench_trace_collection/deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu
OUT_ROOT="$REPO/simulation/examples/tpu_bw_sweep_deepseekv2_maxtext"
SUMMARY="$OUT_ROOT/tpu_bandwidth_sweep_summary.tsv"

TPU_CHIPS=8
TPU_TORUS_DIMS=2,4
LATENCY_NS=100
COLLECTIVE_ALGO=ring
ITERATION_INDEX=1
FALLBACK_COMM_SIZE=$((256 * 1024 * 1024))
COMPUTE_MODEL=kernels
BANDWIDTHS=(200 400 800 1600 3200 6400)

mkdir -p "$OUT_ROOT"
printf "bandwidth_GBps\tstep_ms\tcomm_fraction_pct\n" > "$SUMMARY"

echo "=== TPU Bandwidth Sweep: deepseek-v2-16b MaxText TP2 EP4 DP1 ==="
echo "Trace dir: $TRACE_DIR"
echo "Selected iteration index: $ITERATION_INDEX"
echo "Compute model: $COMPUTE_MODEL"
echo "TPU torus dims: $TPU_TORUS_DIMS"
echo "Fallback comm size: $FALLBACK_COMM_SIZE bytes"
echo

PREV_OUTDIR=""
for BW in "${BANDWIDTHS[@]}"; do
    OUTDIR="$OUT_ROOT/bw${BW}"
    LOG="$OUTDIR/sweep_stdout.log"
    REUSE_ARGS=()
    if [ -n "$PREV_OUTDIR" ]; then
        REUSE_ARGS=(--reuse-et-from "$PREV_OUTDIR")
    fi

    echo "--- TPU ICI bandwidth ${BW} GB/s ---"
    mkdir -p "$OUTDIR"
    python "$REPO/simulation/pipeline.py" \
        --mode tpu-xla \
        --trace-dir "$TRACE_DIR" \
        --output-dir "$OUTDIR" \
        "${REUSE_ARGS[@]}" \
        --compute-model "$COMPUTE_MODEL" \
        --tpu-ranks "$TPU_CHIPS" \
        --tpu-torus-dims "$TPU_TORUS_DIMS" \
        --tpu-iteration-index "$ITERATION_INDEX" \
        --fallback-comm-size "$FALLBACK_COMM_SIZE" \
        --bandwidth "$BW" \
        --latency "$LATENCY_NS" \
        --collective-algo "$COLLECTIVE_ALGO" \
        2>&1 | tee "$LOG"

    RESULT_LINE=$(grep -E "Simulated step time:" "$LOG" | tail -n 1 || true)
    if [ -n "$RESULT_LINE" ]; then
        STEP_MS=$(awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$RESULT_LINE")
        COMM_PCT=$(awk -F'Comm fraction: ' '{print $2}' <<< "$RESULT_LINE" | awk '{gsub("%", "", $1); print $1}')
        printf "%s\t%s\t%s\n" "$BW" "$STEP_MS" "$COMM_PCT" >> "$SUMMARY"
    else
        echo "  WARNING: no pipeline summary found in $LOG"
    fi

    PREV_OUTDIR="$OUTDIR"
    echo
done

echo "=== TPU Bandwidth Summary ==="
awk '
BEGIN { FS = OFS = "\t" }
NR == 1 { next }
NR == 2 {
    base_bw = $1
    base_step = $2
    printf "%12s  %12s  %12s  %12s\n", "BW(GB/s)", "Step(ms)", "Delta(ms)", "Comm(%)"
}
NR >= 2 {
    delta = base_step - $2
    printf "%12s  %12.1f  %12.1f  %11.1f%%\n", $1, $2, delta, $3
    last_bw = $1
    last_step = $2
}
END {
    if (NR >= 2) {
        total_delta = base_step - last_step
        improve = 100.0 * total_delta / base_step
        printf "\n"
        printf "Main readout: %.2fx TPU ICI bandwidth (%s -> %s GB/s) changes simulated step by %.1f ms (%.2f%%).\n", last_bw / base_bw, base_bw, last_bw, total_delta, improve
        printf "Summary TSV: %s\n", summary_path
    }
}
' summary_path="$SUMMARY" "$SUMMARY"
