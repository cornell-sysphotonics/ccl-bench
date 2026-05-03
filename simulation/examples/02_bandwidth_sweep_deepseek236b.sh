#!/bin/bash
# Hardware combo sweep for the DeepSeek V3 236B torchtitan trace.
#
# Each scenario varies scale-up domain size, scale-up bandwidth, and scale-out
# bandwidth together. Uses comm-only kernels+rank replay: non-NCCL GPU kernels
# and measured rank-local idle gaps are fixed, while COMM_COLL_NODE time is
# re-simulated under each hardware configuration.

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-236b-torchtitan-ep32-dp8-pp8-tp4-perlmutter

INTRA_TOPOLOGY=FullyConnected
INTRA_LAT=50
INTER_TOPOLOGY=Switch
INTER_LAT=50000
COLLECTIVE_ALGO=ring
COMPUTE_MODEL=kernels
KERNEL_DEPENDENCY_MODE=rank
ET_WORKERS=4

SCENARIOS=(
    "baseline 4 300 25"
    "balanced 32 1800 50"
    "aggressive 128 3600 100"
)

echo "=== Hardware Combo Sweep: deepseek-v3-236b ep32-dp8-pp8-tp4 ==="
echo "Trace: $TRACE"
echo "Mode: comm-only, compute-model=${COMPUTE_MODEL}, kernel-dependency-mode=${KERNEL_DEPENDENCY_MODE}"
echo "Chakra ET workers: ${ET_WORKERS}"
echo "Scale-up topology: ${INTRA_TOPOLOGY}, latency=${INTRA_LAT} ns"
echo "Scale-out topology: ${INTER_TOPOLOGY}, latency=${INTER_LAT} ns"
echo

SUMMARY="$REPO/simulation/examples/combo_sweep_deepseek236b_kernels/combo_sweep_summary.tsv"
mkdir -p "$(dirname "$SUMMARY")"
printf "scenario\tgpus_per_node\tintra_bandwidth_GBps\tinter_bandwidth_GBps\tstep_ms\tcomm_fraction_pct\n" > "$SUMMARY"

for SCENARIO in "${SCENARIOS[@]}"; do
    read -r NAME GPUS_PER_NODE INTRA_BW INTER_BW <<< "$SCENARIO"
    OUTDIR="$REPO/simulation/examples/combo_sweep_deepseek236b_kernels/${NAME}_su${GPUS_PER_NODE}_intra${INTRA_BW}_inter${INTER_BW}"
    LOG="$OUTDIR/sweep_stdout.log"

    echo "--- ${NAME}: su=${GPUS_PER_NODE}, intra=${INTRA_BW} GB/s, inter=${INTER_BW} GB/s ---"
    mkdir -p "$OUTDIR"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$TRACE" \
        --output-dir "$OUTDIR" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --intra-topology "$INTRA_TOPOLOGY" \
        --intra-bandwidth "$INTRA_BW" \
        --intra-latency "$INTRA_LAT" \
        --topology "$INTER_TOPOLOGY" \
        --bandwidth "$INTER_BW" \
        --latency "$INTER_LAT" \
        --collective-algo "$COLLECTIVE_ALGO" \
        --compute-model "$COMPUTE_MODEL" \
        --kernel-dependency-mode "$KERNEL_DEPENDENCY_MODE" \
        --et-workers "$ET_WORKERS" \
        2>&1 | tee "$LOG"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET|Hardware:" "$LOG" || true

    RESULT_LINE=$(grep -E "Simulated step time:" "$LOG" | tail -n 1 || true)
    if [ -n "$RESULT_LINE" ]; then
        STEP_MS=$(awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$RESULT_LINE")
        COMM_PCT=$(awk -F'Comm fraction: ' '{print $2}' <<< "$RESULT_LINE" | awk '{gsub("%", "", $1); print $1}')
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$GPUS_PER_NODE" "$INTRA_BW" "$INTER_BW" "$STEP_MS" "$COMM_PCT" >> "$SUMMARY"
    fi

    echo
done

echo "=== Hardware Combo Summary ==="
awk '
BEGIN {
    FS = OFS = "\t"
}
NR == 1 {
    next
}
NR == 2 {
    base_name = $1
    base_step = $5
    printf "%12s  %8s  %12s  %12s  %12s  %12s  %12s\n", "Scenario", "SU", "IntraBW", "InterBW", "Step(ms)", "Delta(ms)", "Comm(%)"
}
NR >= 2 {
    delta = base_step - $5
    printf "%12s  %8s  %12s  %12s  %12.1f  %12.1f  %11.1f%%\n", $1, $2, $3, $4, $5, delta, $6
    last_name = $1
    last_step = $5
}
END {
    if (NR >= 2) {
        total_delta = base_step - last_step
        improve = 100.0 * total_delta / base_step
        printf "\n"
        printf "Main readout: %s -> %s changes simulated step by %.1f ms (%.2f%%).\n", base_name, last_name, total_delta, improve
        printf "Interpretation: kernels+rank replay keeps measured kernel/gap timing fixed while re-simulating communication under each hardware combo.\n"
        printf "Summary TSV: %s\n", summary_path
    }
}
' summary_path="$SUMMARY" "$SUMMARY"
