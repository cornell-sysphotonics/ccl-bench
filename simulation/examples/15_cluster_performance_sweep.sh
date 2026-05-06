#!/bin/bash
# Baseline performance sweep for all standard workloads across current/public
# AI networking clusters. Uses the same GPU comm-only toolchain/settings as
# 13_utility_calculation.sh, but does not run any doubling what-ifs.

set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
OUT_ROOT="$REPO/simulation/examples/cluster_performance_sweep"
SUMMARY="$OUT_ROOT/cluster_performance_summary.tsv"
SPECS="$OUT_ROOT/cluster_specs.tsv"

INTRA_LAT=50
INTER_LAT=50000
COLLECTIVE_ALGO=ring
COMPUTE_MODEL=kernels
KERNEL_DEPENDENCY_MODE=rank

mkdir -p "$OUT_ROOT"
printf "cluster_id\tvendor\tplatform\tscaleup_domain\tintra_topology\tinter_topology\tintra_GBps\tinter_GBps\tintra_source\tinter_source\tnotes\n" > "$SPECS"
printf "workload_id\tmodel\tphase\tbatch_or_seq\ttrace\tcluster_id\tplatform\tscaleup_domain\tintra_topology\tinter_topology\tinter_GBps\tintra_GBps\tstep_ms\tcomm_pct\n" > "$SUMMARY"

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
    [ -n "$result_line" ] || return 1
    awk '{for (i = 1; i <= NF; i++) if ($i == "time:") print $(i + 1)}' <<< "$result_line"
}

extract_comm_pct() {
    local log="$1"
    local result_line
    result_line=$(grep -E "Simulated step time:" "$log" | tail -n 1 || true)
    [ -n "$result_line" ] || return 1
    awk -F'Comm fraction: ' '{print $2}' <<< "$result_line" | awk '{gsub("%", "", $1); print $1}'
}

detect_workload_size() {
    local trace="$1"
    find "$trace" -maxdepth 1 -type f -name 'rank*_trace.json' | wc -l
}

declare -a CLUSTER_IDS=()
declare -A CLUSTER_PLATFORMS=()
declare -A CLUSTER_SCALEUP=()
declare -A CLUSTER_INTRA_TOPOLOGY=()
declare -A CLUSTER_INTER_TOPOLOGY=()
declare -A CLUSTER_INTRA=()
declare -A CLUSTER_INTER=()

register_cluster_GBps() {
    local cluster_id="$1"
    local vendor="$2"
    local platform="$3"
    local scaleup_domain="$4"
    local intra_topology="$5"
    local inter_topology="$6"
    local intra_GBps="$7"
    local inter_GBps="$8"
    local intra_source="$9"
    local inter_source="${10}"
    local notes="${11}"

    CLUSTER_IDS+=("$cluster_id")
    CLUSTER_PLATFORMS["$cluster_id"]="$platform"
    CLUSTER_SCALEUP["$cluster_id"]="$scaleup_domain"
    CLUSTER_INTRA_TOPOLOGY["$cluster_id"]="$intra_topology"
    CLUSTER_INTER_TOPOLOGY["$cluster_id"]="$inter_topology"
    CLUSTER_INTRA["$cluster_id"]="$intra_GBps"
    CLUSTER_INTER["$cluster_id"]="$inter_GBps"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$cluster_id" "$vendor" "$platform" "$scaleup_domain" "$intra_topology" "$inter_topology" \
        "$intra_GBps" "$inter_GBps" \
        "$intra_source" "$inter_source" "$notes" >> "$SPECS"
}

register_cluster_mixed() {
    local cluster_id="$1"
    local vendor="$2"
    local platform="$3"
    local scaleup_domain="$4"
    local intra_topology="$5"
    local inter_topology="$6"
    local intra_GBps="$7"
    local inter_Gbps="$8"
    local intra_source="$9"
    local inter_source="${10}"
    local notes="${11}"

    register_cluster_GBps "$cluster_id" "$vendor" "$platform" "$scaleup_domain" \
        "$intra_topology" "$inter_topology" "$intra_GBps" "$(gbps_to_GBps "$inter_Gbps")" \
        "$intra_source" "$inter_source; converted from ${inter_Gbps} Gb/s" "$notes"
}

run_pipeline() {
    local trace="$1"
    local outdir="$2"
    local reuse_from="$3"
    local scaleup_domain="$4"
    local intra_topology="$5"
    local inter_topology="$6"
    local inter_bw="$7"
    local intra_bw="$8"
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
        --gpus-per-node "$scaleup_domain" \
        --intra-topology "$intra_topology" \
        --intra-bandwidth "$intra_bw" \
        --intra-latency "$INTRA_LAT" \
        --topology "$inter_topology" \
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
    local workload_slug reuse_from workload_size
    workload_slug=$(slugify "${workload_id}_${model}_${phase}")
    reuse_from=""

    echo "=== ${workload_id}: ${model} ${phase} (${batch_or_seq}) ==="
    echo "Trace: $trace"
    [ -d "$trace" ] || { echo "ERROR: trace directory not found: $trace" >&2; return 1; }
    workload_size=$(detect_workload_size "$trace")
    [ "$workload_size" -gt 0 ] || { echo "ERROR: no rank*_trace.json files found: $trace" >&2; return 1; }
    echo "Workload size: $workload_size ranks"

    for cluster_id in "${CLUSTER_IDS[@]}"; do
        local platform cluster_scaleup_domain scaleup_domain intra_topology inter_topology inter_bw intra_bw outdir step_ms comm_pct
        platform="${CLUSTER_PLATFORMS[$cluster_id]}"
        cluster_scaleup_domain="${CLUSTER_SCALEUP[$cluster_id]}"
        scaleup_domain="$cluster_scaleup_domain"
        if [ "$scaleup_domain" -gt "$workload_size" ]; then
            scaleup_domain="$workload_size"
        fi
        intra_topology="${CLUSTER_INTRA_TOPOLOGY[$cluster_id]}"
        inter_topology="${CLUSTER_INTER_TOPOLOGY[$cluster_id]}"
        inter_bw="${CLUSTER_INTER[$cluster_id]}"
        intra_bw="${CLUSTER_INTRA[$cluster_id]}"
        outdir="$OUT_ROOT/$workload_slug/$(slugify "$cluster_id")"

        echo "--- $cluster_id: $platform ---"
        echo "Scale-up domain: $scaleup_domain (cluster=$cluster_scaleup_domain, workload=$workload_size), scale-up: $intra_topology/$intra_bw GB/s, scale-out: $inter_topology/$inter_bw GB/s"
        run_pipeline "$trace" "$outdir" "$reuse_from" "$scaleup_domain" "$intra_topology" "$inter_topology" "$inter_bw" "$intra_bw"
        [ -n "$reuse_from" ] || reuse_from="$outdir"

        step_ms=$(extract_step_ms "$outdir/simulation.log")
        comm_pct=$(extract_comm_pct "$outdir/simulation.log")
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$workload_id" "$model" "$phase" "$batch_or_seq" "$trace" \
            "$cluster_id" "$platform" "$scaleup_domain" "$intra_topology" "$inter_topology" "$inter_bw" "$intra_bw" \
            "$step_ms" "$comm_pct" >> "$SUMMARY"
        echo
    done
}

register_cluster_mixed "perlmutter_a100" "NERSC/HPE/NVIDIA" "Perlmutter A100 + Slingshot 11" \
    4 Ring Switch 300 200 "4 A100 NVLink mesh; workload-card 2400 Gb/s converted" "200 Gb/s Slingshot" "Physical scale-out is Slingshot dragonfly; simulated as Switch."
register_cluster_mixed "nvidia_dgx_h200" "NVIDIA" "DGX H200" \
    8 Switch Switch 450 400 "900 GB/s NVSwitch peer-to-peer" "8 x 400 Gb/s ConnectX-7" "Aggregate NIC bandwidth is used as scale-out."
register_cluster_mixed "nvidia_gb200_nvl72" "NVIDIA" "GB200 NVL72 + X800" \
    64 Switch Switch 900 800 "72-GPU NVLink/NVSwitch domain; 1.8 TB/s GPU-to-GPU" "800 Gb/s ConnectX-8/Quantum-X800" "Standalone NVIDIA rack/network spec."
register_cluster_mixed "google_tpu_v6e" "Google" "Cloud TPU v6e" \
    256 Ring Switch 800 200 "2D torus ICI approximated as Ring; 800 GB/s per chip" "4 x 200 Gb/s host NIC" "GPU NCCL traces replayed against TPU-like bandwidth/domain."
register_cluster_mixed "google_tpu7x_ironwood" "Google" "TPU7x Ironwood" \
    9216 Ring Switch 600 100 "3D torus cube approximated as Ring; inferred 1200 GB/s per chip" "800 Gb/s placeholder" "Uses 64-chip cube as scale-up domain."
register_cluster_mixed "google_tpu8t" "Google" "TPU 8t" \
    9600 Ring Switch 1600 400 "ICI topology not yet in Cloud docs; approximated as Ring at 2400 GB/s" "1600 Gb/s placeholder" "Current 2026 training TPU; marked assumption."
register_cluster_mixed "amd_mi355x_pollara400" "AMD" "MI355X UBB + Pollara 400" \
    8 FullyConnected Switch 77 400 "Fully meshed Infinity Fabric; 77 GB/s peer-to-peer I/O" "8 x 400 Gb/s Pollara 400" "State-of-the-art AMD documented platform."
register_cluster_mixed "aws_p6e_gb200" "AWS/NVIDIA" "EC2 P6e-GB200 UltraServer" \
    64 Switch Switch 900 400 "72 GPUs under NVLink/NVSwitch; 1.8 TB/s peer-to-peer" "28.8 Tb/s total EFAv4" "AWS state-of-the-art published UltraServer."

echo "=== Cluster Performance Sweep ==="
echo "Summary TSV: $SUMMARY"
echo "Spec TSV: $SPECS"
echo "Mode: comm-only, compute-model=$COMPUTE_MODEL, kernel-dependency=$KERNEL_DEPENDENCY_MODE"
echo

# run_workload "WL1" "Qwen3-4B" "Inference" "batch=128 input=1024" \
#     "/data/ccl-bench_trace_collection/qwen3-4b-vllm-tp4-batch128-nccl-perlmutter"
# run_workload "WL2" "Llama-3.1-8B" "Inference" "batch=128 input=1024" \
#     "/data/ccl-bench_trace_collection/llama-3.1-8b-vllm-tp4-batch128-perlmutter"
# run_workload "WL3" "DeepSeek-MoE-16B" "Inference" "batch=128 input=1024" \
#     "/data/ccl-bench_trace_collection/deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter"
# run_workload "WL4" "Llama-3.1-8B" "Training" "batch=4 sequence=512" \
#     "/data/ccl-bench_trace_collection/llama3_8b_wl4_fsdp2_tp4_batch4_seq512-nccl-perlmutter"
# run_workload "WL5" "DeepSeek-V3-16B" "Training" "batch=8 sequence=1024" \
#     "/data/ccl-bench_trace_collection/deepseek_v3_16b_w8_batch8_seq1024-nccl-perlmutter"
run_workload "WL6" "DeepSeek-V3-16B" "Training" "batch=64 sequence=2048" \
    "/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter"
run_workload "WL7" "DeepSeek-V3-236B" "Training" "batch=64 sequence=1024" \
    "/data/ccl-bench_trace_collection/deepseek-v3-236b-torchtitan-ep32-dp8-pp8-tp4-perlmutter"

echo "=== Cluster Performance Summary ==="
awk '
BEGIN { FS = OFS = "\t" }
NR == 1 { next }
{
    printf "%-4s %-20s %-9s %-24s intra=%-14s inter=%-8s step=%10s ms  comm=%7s%%\n", $1, $2, $3, $6, $9, $10, $13, $14
}
' "$SUMMARY"
echo
echo "Summary TSV: $SUMMARY"
echo "Spec TSV: $SPECS"
