#!/bin/bash
# TPU scale-up utility score for selected standard workloads.
#
# Runs one baseline and one 2x scale-up bandwidth what-if per workload.
# Utility score is percent step-time improvement per doubling:
#   100 * (baseline_step_ms - scaleup_2x_step_ms) / baseline_step_ms

set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT_ROOT="$REPO/simulation/examples/tpu_scaleup_utility"
SUMMARY="$OUT_ROOT/tpu_scaleup_utility_summary.tsv"

TPU_CHIPS=8
TPU_RING_DIMS=8
LATENCY_NS=100
COLLECTIVE_ALGO=ring
COMPUTE_MODEL=kernels
FALLBACK_COMM_SIZE=$((256 * 1024 * 1024))

mkdir -p "$OUT_ROOT"
printf "workload_id\tmodel\tphase\tbatch_or_seq\ttrace_dir\ttrace_json\tbase_scaleup_GBps\tscaleup_domain\tbase_step_ms\tbase_comm_pct\tscaleup_2x_step_ms\tscaleup_utility_pct\n" > "$SUMMARY"

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

run_tpu_pipeline() {
    local trace_dir="$1"
    local trace_json="$2"
    local outdir="$3"
    local reuse_from="$4"
    local bandwidth="$5"
    local iteration_marker="$6"
    local iteration_index="$7"
    local min_iteration_duration_us="$8"

    local reuse_args=()
    if [ -n "$reuse_from" ]; then
        reuse_args=(--reuse-et-from "$reuse_from")
    fi

    mkdir -p "$outdir"
    python "$REPO/simulation/pipeline.py" \
        --mode tpu-xla \
        --trace-dir "$trace_dir" \
        --tpu-trace-json "$trace_json" \
        --output-dir "$outdir" \
        "${reuse_args[@]}" \
        --compute-model "$COMPUTE_MODEL" \
        --tpu-ranks "$TPU_CHIPS" \
        --tpu-torus-dims "$TPU_RING_DIMS" \
        --tpu-iteration-marker "$iteration_marker" \
        --tpu-iteration-index "$iteration_index" \
        --tpu-min-iteration-duration-us "$min_iteration_duration_us" \
        --fallback-comm-size "$FALLBACK_COMM_SIZE" \
        --bandwidth "$bandwidth" \
        --latency "$LATENCY_NS" \
        --collective-algo "$COLLECTIVE_ALGO" \
        2>&1 | tee "$outdir/simulation.log"
    grep -E "Selected iteration|Simulated step|[Cc]omm fraction|ERROR|Hardware:" "$outdir/simulation.log" || true
}

run_workload() {
    local workload_id="$1"
    local model="$2"
    local phase="$3"
    local batch_or_seq="$4"
    local trace_dir="$5"
    local trace_json="$6"
    local base_scaleup_bw="$7"
    local iteration_marker="$8"
    local iteration_index="$9"
    local min_iteration_duration_us="${10}"

    local double_scaleup_bw slug base_out scaleup_out
    double_scaleup_bw=$(awk -v bw="$base_scaleup_bw" 'BEGIN { printf "%.6g", bw * 2.0 }')
    slug=$(slugify "${workload_id}_${model}_${phase}")
    base_out="$OUT_ROOT/$slug/base"
    scaleup_out="$OUT_ROOT/$slug/scaleup_bw_2x"

    echo "=== ${workload_id}: ${model} ${phase} (${batch_or_seq}) ==="
    echo "Trace dir: $trace_dir"
    echo "Trace JSON: $trace_json"
    echo "Scale-up domain: ${TPU_CHIPS}, topology: Ring, baseline ICI: ${base_scaleup_bw} GB/s"
    echo

    if [ ! -d "$trace_dir" ]; then
        echo "ERROR: trace directory not found: $trace_dir" >&2
        return 1
    fi

    run_tpu_pipeline "$trace_dir" "$trace_json" "$base_out" "" \
        "$base_scaleup_bw" "$iteration_marker" "$iteration_index" "$min_iteration_duration_us"
    run_tpu_pipeline "$trace_dir" "$trace_json" "$scaleup_out" "$base_out" \
        "$double_scaleup_bw" "$iteration_marker" "$iteration_index" "$min_iteration_duration_us"

    local base_step base_comm scaleup_step scaleup_score
    base_step=$(extract_step_ms "$base_out/simulation.log")
    base_comm=$(extract_comm_pct "$base_out/simulation.log")
    scaleup_step=$(extract_step_ms "$scaleup_out/simulation.log")
    scaleup_score=$(utility_pct "$base_step" "$scaleup_step")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$workload_id" "$model" "$phase" "$batch_or_seq" \
        "$trace_dir" "$trace_json" "$base_scaleup_bw" "$TPU_CHIPS" \
        "$base_step" "$base_comm" "$scaleup_step" "$scaleup_score" >> "$SUMMARY"
    echo
}

echo "=== TPU Scale-up Utility Calculation ==="
echo "Summary TSV: $SUMMARY"
echo "Topology: Ring, scale-up domain: $TPU_CHIPS TPU chips"
echo "Score: percent step-time improvement per 2x scale-up bandwidth"
echo

# run_workload "WL1" "Qwen3-4B" "Inference" "batch=128 input=1024" \
#     "/data/ccl-bench_trace_collection/Qwen3-4B-torchxla-vllm-tp8-tpu-group-4" \
#     "Qwen3-4B-torchxla-vllm-tp8-batch-128-tpu-group-4.json" \
#     800 "jit_run_model" 0 0

# run_workload "WL2" "Llama-3.1-8B" "Inference" "batch=128 input=1024" \
#     "/data/ccl-bench_trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4" \
#     "Llama-3.1-8B-torchxla-vllm-tp8-batch-128-tpu-group-4.json" \
#     800 "jit_run_model" 0 0

# run_workload "WL4" "Llama-3.1-8B" "Training" "batch=8 sequence=512" \
#     "/data/ccl-bench_trace_collection/llama-3.1-8b-torchxla_fsdp_v6e-8-tpu-group_21" \
#     "llama-3.1-8b-torchxla_fsdp_v6e-8-tpu-group_21.trace.json" \
#     6400 "SyncTensorsGraph" 1 1000000

run_workload "WL5" "DeepSeek-V2-16B" "Training" "batch=8 sequence=1024" \
    "/data/ccl-bench_trace_collection/deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu" \
    "t1v-n-975c9bdd-w-0.trace.json" \
    3200 "jit_train_step" 0 100000

echo "=== TPU Scale-up Utility Summary ==="
awk '
BEGIN { FS = OFS = "\t" }
NR == 1 { next }
{
    printf "%-4s %-20s %-9s scale-up=%8.4f%%  base=%s ms  2x=%s ms\n", $1, $2, $3, $12, $9, $11
}
' "$SUMMARY"
echo
echo "Summary TSV: $SUMMARY"
