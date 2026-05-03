#!/bin/bash
# Bandwidth sweep: how does simulated step time and comm fraction change as we
# scale inter-node network bandwidth from 1.25 GB/s to 200 GB/s?
#
# The final summary emphasizes sensitivity relative to the slowest-bandwidth run.
# In comm-only gap mode, replayed COMP_NODE gaps are fixed; only exposed
# COMM_COLL_NODE time changes with --bandwidth.

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
# TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep8-dp2-tp4-perlmutter
# TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter
GPUS_PER_NODE=4
INTRA_BW=300
INTRA_LAT=25

echo "=== Scale-out Bandwidth Sweep: deepseek-v3-16b ep8-dp2-tp4 ==="
echo "Mode: comm-only, compute-model=gap"
echo "Fixed scale-up: Ring, ${INTRA_BW} GB/s, ${INTRA_LAT} ns"
echo

SUMMARY="$REPO/simulation/examples/bw_sweep_kernels/bandwidth_sweep_summary.tsv"
mkdir -p "$(dirname "$SUMMARY")"
printf "bandwidth_GBps\tstep_ms\tcomm_fraction_pct\n" > "$SUMMARY"

PREV_OUTDIR="scaleup_bw_sweep_kernels/su_ep32_32_bw112.5_FullyConnected_interSwitch"
# PREV_OUTDIR="scaleup_bw_sweep_kernels/su_ep8_8_bw112.5_FullyConnected_interSwitch"
for BW in 1.25 5 12.5 25 50 100 200; do
    # OUTDIR="$REPO/simulation/examples/ccl_bench_sim_bw${BW}_ep8"
    # OUTDIR="$REPO/simulation/examples/ccl_bench_sim_bw${BW}_ep4_kernels"
    OUTDIR="$REPO/simulation/examples/bw_sweep_kernels/bw_sweep_ep32_${BW}"
    # OUTDIR="$REPO/simulation/examples/bw_sweep_kernels/bw_sweep_ep8_${BW}"
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
        --compute-model kernels \
        --kernel-dependency-mode rank \
        2>&1 | tee "$LOG"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET" "$LOG" || true

    RESULT_LINE=$(grep -E "Simulated step time:" "$LOG" | tail -n 1 || true)
    if [ -n "$RESULT_LINE" ]; then
        STEP_MS=$(awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$RESULT_LINE")
        COMM_PCT=$(awk -F'Comm fraction: ' '{print $2}' <<< "$RESULT_LINE" | awk '{gsub("%", "", $1); print $1}')
        printf "%s\t%s\t%s\n" "$BW" "$STEP_MS" "$COMM_PCT" >> "$SUMMARY"
    fi

    PREV_OUTDIR="$OUTDIR"
    echo
done

echo "=== Bandwidth Sensitivity Summary ==="
awk '
BEGIN {
    FS = OFS = "\t"
}
NR == 1 {
    next
}
NR == 2 {
    base_bw = $1
    base_step = $2
    printf "%12s  %12s  %12s  %12s  %12s\n", "BW(GB/s)", "Step(ms)", "Delta(ms)", "Improve(%)", "Comm(%)"
}
NR >= 2 {
    delta = base_step - $2
    improve = 100.0 * delta / base_step
    printf "%12s  %12.1f  %12.1f  %11.2f%%  %11.1f%%\n", $1, $2, delta, improve, $3
    last_bw = $1
    last_step = $2
    last_improve = improve
}
	END {
	    if (NR >= 2) {
	        total_delta = base_step - last_step
	        printf "\n"
	        printf "Main readout: %.2fx inter-node bandwidth (%s -> %s GB/s) changes simulated step by %.1f ms (%.2f%%).\n", last_bw / base_bw, base_bw, last_bw, total_delta, last_improve
	        printf "Interpretation: kernels+rank replay keeps measured kernel/gap timing fixed while re-simulating COMM_COLL_NODE time under each bandwidth.\n"
	        printf "Summary TSV: %s\n", summary_path
	    }
	}
' summary_path="$SUMMARY" "$SUMMARY"
