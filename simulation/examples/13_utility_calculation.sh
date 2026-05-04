#!/bin/bash
# Utility score for representative standard workloads.
#
# For each workload this runs a baseline simulation plus one 2x what-if for:
#   1. scale-out bandwidth
#   2. scale-up bandwidth
#   3. scale-up domain size
#
# Utility score is reported as percent step-time improvement per doubling:
#   100 * (baseline_step_ms - variant_step_ms) / baseline_step_ms
#
# Bandwidths below mirror the current workload-card values and are converted
# from Gb/s to GB/s for simulation/pipeline.py. Once card parsing is available,
# replace INTER_BW_GBPS and INTRA_BW_GBPS with values read from each card.

set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT_ROOT="$REPO/simulation/examples/utility_calculation"
SUMMARY="$OUT_ROOT/utility_summary.tsv"

INTRA_TOPOLOGY=Ring
INTER_TOPOLOGY=Switch
INTRA_LAT=50
INTER_LAT=50000
COLLECTIVE_ALGO=ring
COMPUTE_MODEL=kernels
KERNEL_DEPENDENCY_MODE=rank

mkdir -p "$OUT_ROOT"
printf "workload_id\tmodel\tphase\tbatch_or_seq\ttrace\tbase_inter_GBps\tbase_intra_GBps\tbase_scaleup_domain\tbase_step_ms\tbase_comm_pct\tscaleout_2x_step_ms\tscaleout_utility_pct\tscaleup_bw_2x_step_ms\tscaleup_bw_utility_pct\tscaleup_domain_2x_step_ms\tscaleup_domain_utility_pct\n" > "$SUMMARY"

gbps_to_GBps() {
    awk -v gbps="$1" 'BEGIN { printf "%.6g", gbps / 8.0 }'
}

slugify() {
    tr '[:upper:]' '[:lower:]' <<< "$1" | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//'
}

extract_step_ms() {
    local log="$1"
    local result_line
    result_line=$(grep -E "Simulated step time:" "$log" | tail -n 1 || true)
    if [ -z "$result_line" ]; then
        return 1
    fi
    awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$result_line"
}

extract_comm_pct() {
    local log="$1"
    local result_line
    result_line=$(grep -E "Simulated step time:" "$log" | tail -n 1 || true)
    if [ -z "$result_line" ]; then
        return 1
    fi
    awk -F'Comm fraction: ' '{print $2}' <<< "$result_line" | awk '{gsub("%", "", $1); print $1}'
}

utility_pct() {
    local base_step="$1"
    local variant_step="$2"
    awk -v base="$base_step" -v variant="$variant_step" 'BEGIN { printf "%.4f", 100.0 * (base - variant) / base }'
}

run_pipeline() {
    local trace="$1"
    local outdir="$2"
    local reuse_from="$3"
    local gpus_per_node="$4"
    local inter_bw="$5"
    local intra_bw="$6"

    local reuse_args=()
    if [ -n "$reuse_from" ]; then
        reuse_args=(--reuse-et-from "$reuse_from")
    fi

    mkdir -p "$outdir"
    python "$REPO/simulation/pipeline.py" \
        --mode comm-only \
        --trace-dir "$trace" \
        --output-dir "$outdir" \
        "${reuse_args[@]}" \
        --gpus-per-node "$gpus_per_node" \
        --intra-topology "$INTRA_TOPOLOGY" \
        --intra-bandwidth "$intra_bw" \
        --intra-latency "$INTRA_LAT" \
        --topology "$INTER_TOPOLOGY" \
        --bandwidth "$inter_bw" \
        --latency "$INTER_LAT" \
        --collective-algo "$COLLECTIVE_ALGO" \
        --compute-model "$COMPUTE_MODEL" \
        --kernel-dependency-mode "$KERNEL_DEPENDENCY_MODE" \
        2>&1 | tee "$outdir/simulation.log"
    grep -E "Simulated step|[Cc]omm fraction|ERROR|Reused .*Chakra ET|Generating Chakra ET|Hardware:" "$outdir/simulation.log" || true
}

run_workload() {
    local workload_id="$1"
    local model="$2"
    local phase="$3"
    local batch_or_seq="$4"
    local trace="$5"
    local inter_bw_gbps="$6"
    local intra_bw_gbps="$7"
    local scaleup_domain="$8"

    local inter_bw intra_bw double_inter_bw double_intra_bw double_scaleup_domain slug
    inter_bw=$(gbps_to_GBps "$inter_bw_gbps")
    intra_bw=$(gbps_to_GBps "$intra_bw_gbps")
    double_inter_bw=$(awk -v bw="$inter_bw" 'BEGIN { printf "%.6g", bw * 2.0 }')
    double_intra_bw=$(awk -v bw="$intra_bw" 'BEGIN { printf "%.6g", bw * 2.0 }')
    double_scaleup_domain=$((scaleup_domain * 2))
    slug=$(slugify "${workload_id}_${model}_${phase}")

    echo "=== ${workload_id}: ${model} ${phase} (${batch_or_seq}) ==="
    echo "Trace: $trace"
    echo "Baseline: scale-out ${inter_bw} GB/s, scale-up ${intra_bw} GB/s, scale-up domain ${scaleup_domain}"
    echo

    if [ ! -d "$trace" ]; then
        echo "ERROR: trace directory not found: $trace" >&2
        return 1
    fi

    local base_out scaleout_out scaleup_bw_out scaleup_domain_out
    base_out="$OUT_ROOT/$slug/base"
    scaleout_out="$OUT_ROOT/$slug/scaleout_bw_2x"
    scaleup_bw_out="$OUT_ROOT/$slug/scaleup_bw_2x"
    scaleup_domain_out="$OUT_ROOT/$slug/scaleup_domain_2x"

    run_pipeline "$trace" "$base_out" "" "$scaleup_domain" "$inter_bw" "$intra_bw"
    run_pipeline "$trace" "$scaleout_out" "$base_out" "$scaleup_domain" "$double_inter_bw" "$intra_bw"
    run_pipeline "$trace" "$scaleup_bw_out" "$base_out" "$scaleup_domain" "$inter_bw" "$double_intra_bw"
    run_pipeline "$trace" "$scaleup_domain_out" "$base_out" "$double_scaleup_domain" "$inter_bw" "$intra_bw"

    local base_step base_comm scaleout_step scaleup_bw_step scaleup_domain_step
    local scaleout_score scaleup_bw_score scaleup_domain_score
    base_step=$(extract_step_ms "$base_out/simulation.log")
    base_comm=$(extract_comm_pct "$base_out/simulation.log")
    scaleout_step=$(extract_step_ms "$scaleout_out/simulation.log")
    scaleup_bw_step=$(extract_step_ms "$scaleup_bw_out/simulation.log")
    scaleup_domain_step=$(extract_step_ms "$scaleup_domain_out/simulation.log")
    scaleout_score=$(utility_pct "$base_step" "$scaleout_step")
    scaleup_bw_score=$(utility_pct "$base_step" "$scaleup_bw_step")
    scaleup_domain_score=$(utility_pct "$base_step" "$scaleup_domain_step")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$workload_id" "$model" "$phase" "$batch_or_seq" "$trace" \
        "$inter_bw" "$intra_bw" "$scaleup_domain" \
        "$base_step" "$base_comm" \
        "$scaleout_step" "$scaleout_score" \
        "$scaleup_bw_step" "$scaleup_bw_score" \
        "$scaleup_domain_step" "$scaleup_domain_score" >> "$SUMMARY"
    echo
}

echo "=== Utility Calculation ==="
echo "Summary TSV: $SUMMARY"
echo "Mode: comm-only, compute-model=$COMPUTE_MODEL, kernel-dependency=$KERNEL_DEPENDENCY_MODE"
echo "Score: percent step-time improvement per doubling"
echo

# Workload-card bandwidths: bandwidth_gbps[0]=scale-out, [1]=scale-up.
# All seven representative traces currently use 200 Gb/s scale-out and
# 2400 Gb/s scale-up in their workload cards.
run_workload "WL1" "Qwen3-4B" "Inference" "batch=128 input=1024" \
    "/data/ccl-bench_trace_collection/qwen3-4b-vllm-tp4-batch128-nccl-perlmutter" \
    200 2400 4
run_workload "WL2" "Llama-3.1-8B" "Inference" "batch=128 input=1024" \
    "/data/ccl-bench_trace_collection/llama-3.1-8b-vllm-tp4-batch128-perlmutter" \
    200 2400 4
run_workload "WL3" "DeepSeek-MoE-16B" "Inference" "batch=128 input=1024" \
    "/data/ccl-bench_trace_collection/deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter" \
    200 2400 4
run_workload "WL4" "Llama-3.1-8B" "Training" "batch=4 sequence=512" \
    "/data/ccl-bench_trace_collection/llama3_8b_wl4_fsdp2_tp4_batch4_seq512-nccl-perlmutter" \
    200 2400 4
run_workload "WL5" "DeepSeek-V3-16B" "Training" "batch=8 sequence=1024" \
    "/data/ccl-bench_trace_collection/deepseek_v3_16b_w8_batch8_seq1024-nccl-perlmutter" \
    200 2400 4
run_workload "WL6" "DeepSeek-V3-16B" "Training" "batch=64 sequence=2048" \
    "/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter" \
    200 2400 4
run_workload "WL7" "DeepSeek-V3-236B" "Training" "batch=64 sequence=1024" \
    "/data/ccl-bench_trace_collection/deepseek-v3-236b-torchtitan-ep32-dp8-pp8-tp4-perlmutter" \
    200 2400 4

echo "=== Utility Summary ==="
awk '
BEGIN {
    FS = OFS = "\t"
}
NR == 1 {
    next
}
{
    printf "%-4s %-20s %-9s scale-out=%8.4f%%  scale-up-bw=%8.4f%%  scale-up-domain=%8.4f%%  base=%s ms\n", $1, $2, $3, $12, $14, $16, $9
}
' "$SUMMARY"
echo
echo "Summary TSV: $SUMMARY"
