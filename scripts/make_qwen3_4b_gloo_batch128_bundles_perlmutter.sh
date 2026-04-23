#!/usr/bin/env bash
set -euo pipefail

# Bundle Qwen3-4B Gloo run into a website-ready card. Mirrors the structure
# of make_qwen3_4b_kineto_comm_bundles_perlmutter.sh but sets
# comm_library.name: Gloo and records the two CCL_BENCH_* env vars that
# actually determined the backend.

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/qwen3-4b-gloo-batch128}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/qwen3-4b-gloo-batch128-bundles}"
TRACE_URL_ROOT="${TRACE_URL_ROOT:-/data/ccl-bench_trace_collection}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-1152}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TP="${TP:-4}"

mkdir -p "$BUNDLE_ROOT"

link_or_copy() {
  local src=$1
  local dst=$2
  ln -f "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
}

write_yaml() {
  local out=$1
  local name=$2
  local tp=$3
  local batch=$4

  cat > "$out" <<YAML
version: 1

description: >
  Qwen3-4B inference, TP=${tp}, batch size ${batch}, communication=Gloo.
  Gloo is torch.distributed's TCP backend; it has no NVLink or RDMA fast path
  and serves as the bandwidth-floor baseline for the ccl-bench comparison.
  Global batch size is ${batch} requests. Traces were collected on Perlmutter
  with vLLM bench latency, PyTorch Kineto profiler, and Nsight Systems.
  Input length is ${INPUT_LEN} and output length is ${OUTPUT_LEN}; seq_len
  records input plus output tokens for MFU accounting.

hf_url: https://huggingface.co/Qwen/Qwen3-4B
trace_url: ${TRACE_URL_ROOT}/${name}
contributor: Kaiwen Guo
contact: kg597@cornell.edu

workload:
  model:
    phase: inference
    moe: false
    granularity: model_fwd
    model_family: qwen3-4b
    precision: bf16
    epochs: 1
    iteration: 1
    model_arch:
      num_params: 4022468096
      num_params_embedding: 388956160
      num_layers: 36
      num_heads: 32
      head_dim: 128
  data:
    batch_size: ${batch}
    batch_size_scope: global
    input_len: ${INPUT_LEN}
    output_len: ${OUTPUT_LEN}
    seq_len: ${SEQ_LEN} # input_len + output_len
    dataset: random_${INPUT_LEN}_input_${OUTPUT_LEN}_output
  hardware:
    network_topo:
      topology: slingshot
      bandwidth_gbps:
        - 200
        - 2400
    xpu_spec:
      type: GPU
      model: nvidia_a100
      total_count: 4
      count_per_node: 4
    driver_version: cuda_12.8

Model-executor:
  framework:
    name: vllm
    version: "0.19.0"
    compiler_tool_selection: plain_pytorch
  model_plan_parallelization:
    dp_replicate: 1
    dp_shard: 1
    tp: ${tp}
    pp: 1
    cp: 1
    ep: 1
    pp_mb: 1
  communication_library:
    name: Gloo
    version: torch-2.10.0
    env:
      CCL_BENCH_DIST_BACKEND: "gloo"
      CCL_BENCH_FORCE_TORCH_DISTRIBUTED: "1"
      VLLM_DISABLE_CUSTOM_ALL_REDUCE: "1"
  protocol_selection:
    - tcp

metric_source:
  traces:
    - nsys
    - json
  metrics_specific_trace:
    - nsys
    - vllm_bench_latency
YAML
}

bundle_one() {
  local tp=$1
  local batch=$2
  local name="qwen3-4b-vllm-tp${tp}-batch${batch}-gloo-perlmutter"
  local src="$SRC_ROOT/$name"
  local dst="$BUNDLE_ROOT/$name"
  local logs="$dst/logs"

  if [[ ! -d "$src" ]]; then
    echo "missing source directory: $src" >&2
    return 1
  fi

  rm -rf "$dst"
  mkdir -p "$dst" "$logs"

  write_yaml "$dst/$name.yaml" "$name" "$tp" "$batch"

  local i=0
  while IFS= read -r trace; do
    link_or_copy "$trace" "$dst/rank${i}_trace.json"
    i=$((i+1))
  done < <(find "$src/kineto" -maxdepth 1 -type f -name "*.pt.trace.json" | sort)

  if (( i == 0 )); then
    echo "no Kineto trace JSON files found in $src/kineto" >&2
    return 1
  fi

  for ext in sqlite nsys-rep latency.json bench.log profile.log; do
    while IFS= read -r f; do
      [[ -e "$f" ]] || continue
      case "$ext" in
        sqlite|nsys-rep) link_or_copy "$f" "$dst/$(basename "$f")" ;;
        *) link_or_copy "$f" "$logs/$(basename "$f")" ;;
      esac
    done < <(find "$src" -maxdepth 1 -type f -name "*.${ext}" 2>/dev/null | sort)
  done

  cat > "$dst/README.md" <<MD
# ${name}

Collected on Perlmutter A100 with \`vllm bench latency\`.

- Input length: ${INPUT_LEN}
- Output length: ${OUTPUT_LEN}
- Batch size: ${batch}
- Tensor parallelism: ${tp}
- Communication library: Gloo (torch.distributed TCP backend, single node)
- Primary metric sources: NSYS SQLite plus PyTorch Kineto JSON
- Additional artifacts: NSYS \`.nsys-rep\`, \`.sqlite\`, and raw vLLM logs under \`logs/\`

## Gloo engagement

This run requires the ccl-bench vLLM patch. Both env vars must be set:

- \`CCL_BENCH_DIST_BACKEND=gloo\`
- \`CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1\`

\`logs/${name}.bench.log\` contains the \`backend=gloo\` line from
\`parallel_state.py\` that confirms Gloo was the active backend. The NSYS
sqlite should contain no \`ncclDevKernel_*\` kernels; collective ops appear
as CPU-side \`send\`/\`recv\` traffic.
MD

  echo "created $dst with $i rank trace(s)"
}

bundle_one "$TP" "$BATCH_SIZE"

echo
echo "Bundle root: $BUNDLE_ROOT"
find "$BUNDLE_ROOT" -maxdepth 2 -type f | sort
