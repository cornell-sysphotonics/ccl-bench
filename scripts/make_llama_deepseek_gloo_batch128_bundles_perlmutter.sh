#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-gloo-batch128}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-gloo-batch128-bundles}"
TRACE_URL_ROOT="${TRACE_URL_ROOT:-/data/ccl-bench_trace_collection}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-1152}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "$BUNDLE_ROOT"

link_or_copy() {
  local src=$1
  local dst=$2
  ln -f "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
}

write_yaml() {
  local out=$1
  local name=$2
  local desc=$3
  local hf_url=$4
  local moe=$5
  local family=$6
  local num_params=$7
  local num_params_embedding=$8
  local layers=$9
  local heads=${10}
  local head_dim=${11}
  local tp=${12}
  local ep=${13}
  local num_params_active=${14:-}
  local active_params_yaml=""
  if [[ -n "$num_params_active" ]]; then
    active_params_yaml="      num_params_active: ${num_params_active}"$'\n'
  fi

  cat > "$out" <<YAML
version: 1

description: >
  ${desc}
  Global batch size is ${BATCH_SIZE} requests. Traces were collected on
  Perlmutter with vLLM bench latency, PyTorch Kineto profiler, and Nsight
  Systems. Input length is ${INPUT_LEN} and output length is ${OUTPUT_LEN};
  seq_len records input plus output tokens for MFU accounting. Gloo was forced
  through torch.distributed by the ccl-bench vLLM patch.

hf_url: ${hf_url}
trace_url: ${TRACE_URL_ROOT}/${name}
contributor: Kaiwen Guo
contact: kg597@cornell.edu

workload:
  model:
    phase: inference
    moe: ${moe}
    granularity: model_fwd
    model_family: ${family}
    precision: bf16
    epochs: 1
    iteration: 1
    model_arch:
      num_params: ${num_params}
${active_params_yaml}
      num_params_embedding: ${num_params_embedding}
      num_layers: ${layers}
      num_heads: ${heads}
      head_dim: ${head_dim}
  data:
    batch_size: ${BATCH_SIZE}
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
    ep: ${ep}
    pp_mb: 1
  communication_library:
    name: Gloo
    version: torch.distributed
    env:
      CCL_BENCH_DIST_BACKEND: gloo
      CCL_BENCH_FORCE_TORCH_DISTRIBUTED: "1"
      NCCL_IB_DISABLE: "1"
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
  local name=$1
  local desc=$2
  local hf_url=$3
  local moe=$4
  local family=$5
  local num_params=$6
  local num_params_embedding=$7
  local layers=$8
  local heads=$9
  local head_dim=${10}
  local tp=${11}
  local ep=${12}
  local num_params_active=${13:-}

  local src="$SRC_ROOT/$name"
  local dst="$BUNDLE_ROOT/$name"
  local logs="$dst/logs"

  if [[ ! -d "$src" ]]; then
    echo "missing source directory: $src" >&2
    return 1
  fi

  rm -rf "$dst"
  mkdir -p "$dst" "$logs"

  write_yaml "$dst/$name.yaml" "$name" "$desc" "$hf_url" "$moe" "$family" \
    "$num_params" "$num_params_embedding" "$layers" "$heads" "$head_dim" \
    "$tp" "$ep" "$num_params_active"

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
- Batch size: ${BATCH_SIZE}
- Tensor parallelism: ${tp}
- Expert parallelism: ${ep}
- Communication library: Gloo through \`torch.distributed\`
- Required vLLM patch: \`scripts/vllm_gloo_patch\`
- Primary metric sources: NSYS SQLite plus PyTorch Kineto JSON
- Additional artifacts: NSYS \`.nsys-rep\`, \`.sqlite\`, and raw vLLM logs under \`logs/\`
MD

  echo "created $dst with $i rank trace(s)"
}

bundle_one \
  "llama-3.1-8b-vllm-tp4-batch${BATCH_SIZE}-gloo-perlmutter" \
  "Llama-3.1-8B inference, TP=4, batch size ${BATCH_SIZE}, communication=Gloo." \
  "https://huggingface.co/meta-llama/Llama-3.1-8B" false llama-3.1-8b \
  8030261248 1050673152 32 32 128 4 1

bundle_one \
  "deepseek-moe-16b-vllm-tp4-ep4-batch${BATCH_SIZE}-gloo-perlmutter" \
  "DeepSeek-MoE-16B inference, TP=4, EP=4, batch size ${BATCH_SIZE}, communication=Gloo." \
  "https://huggingface.co/deepseek-ai/deepseek-moe-16b-base" true deepseek-moe-16b \
  16375728128 419430400 28 16 128 4 4 2828650496

echo
echo "Bundle root: $BUNDLE_ROOT"
find "$BUNDLE_ROOT" -maxdepth 2 -type f | sort
