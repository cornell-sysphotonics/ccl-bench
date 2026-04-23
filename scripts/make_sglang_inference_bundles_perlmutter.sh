#!/usr/bin/env bash
set -uo pipefail

# Bundle SGLang server+bench_serving outputs into ccl-bench trace directories.

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-inference}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-inference-bundles}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-$((INPUT_LEN + OUTPUT_LEN))}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.10.post1}"
COMM_BACKEND="${COMM_BACKEND:-nccl}"

case "$COMM_BACKEND" in
  nccl)
    COMM_LIBRARY_NAME="NCCL"
    COMM_PROTOCOLS=("p2p" "nvlink")
    ;;
  gloo)
    COMM_LIBRARY_NAME="Gloo"
    COMM_PROTOCOLS=("tcp")
    ;;
  mscclpp)
    COMM_LIBRARY_NAME="NCCL+MSCCL++"
    COMM_PROTOCOLS=("p2p" "nvlink")
    ;;
  pure_mscclpp)
    COMM_LIBRARY_NAME="MSCCL++"
    COMM_PROTOCOLS=("p2p" "nvlink")
    ;;
  native_mscclpp)
    COMM_LIBRARY_NAME="SGLang-MSCCL++"
    COMM_PROTOCOLS=("p2p" "nvlink")
    ;;
  *)
    COMM_LIBRARY_NAME="$COMM_BACKEND"
    COMM_PROTOCOLS=("p2p" "nvlink")
    ;;
esac

mkdir -p "$BUNDLE_ROOT"

write_yaml() {
  local out_yaml=$1
  local name=$2
  local desc=$3
  local hf_url=$4
  local family=$5
  local moe=$6
  local batch=$7
  local tp=$8
  local ep=$9
  local num_params=${10}
  local num_emb=${11}
  local layers=${12}
  local heads=${13}
  local head_dim=${14}
  local active_params=${15}

  cat > "$out_yaml" <<EOF
version: 1

description: >
  ${desc}

hf_url: ${hf_url}
trace_url:

workload:
  model:
    phase: inference
    moe: ${moe}
    granularity: model_fwd
    model_family: ${family}
    precision: bf16
    num_params: ${num_params}
    num_params_embedding: ${num_emb}
    num_layers: ${layers}
    num_heads: ${heads}
    head_dim: ${head_dim}
EOF

  if [[ "$active_params" != "0" ]]; then
    cat >> "$out_yaml" <<EOF
    num_params_active: ${active_params}
EOF
  fi

  cat >> "$out_yaml" <<EOF
  data:
    batch_size: ${batch}
    seq_len: ${SEQ_LEN}
    input_len: ${INPUT_LEN}
    output_len: ${OUTPUT_LEN}
    dataset: random
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
    driver_version: cuda_12.x

Model-executor:
  framework:
    name: sglang
    version: ${SGLANG_VERSION}
    compiler_tool_selection: plain_pytorch
    env:
      SGLANG_ENV: "/pscratch/sd/k/kg597/session1/csglang"
      DISABLE_CUDA_GRAPH: "1"
      DISABLE_PIECEWISE_CUDA_GRAPH: "1"
      DISABLE_CUSTOM_ALL_REDUCE: "1"
      DATASET_NAME: "random-ids"
  model_plan_parallelization:
    dp_replicate: 1
    dp_shard: 1
    tp: ${tp}
    pp: 1
    cp: 1
EOF

  if [[ "$ep" != "1" ]]; then
    cat >> "$out_yaml" <<EOF
    ep: ${ep}
EOF
  fi

  cat >> "$out_yaml" <<EOF
  communication_library:
    name: ${COMM_LIBRARY_NAME}
    env:
      NCCL_IB_DISABLE: "1"
  protocol_selection:
EOF

  for protocol in "${COMM_PROTOCOLS[@]}"; do
    cat >> "$out_yaml" <<EOF
    - ${protocol}
EOF
  done

  cat >> "$out_yaml" <<EOF

metric_source:
  traces:
    - sglang
EOF

  if [[ -f "$SRC_ROOT/$name/${name}.sqlite" ]]; then
    cat >> "$out_yaml" <<EOF
    - nsys
EOF
  fi

  cat >> "$out_yaml" <<EOF
  metrics_specific_trace:
    - sglang_bench_serving
EOF

  if [[ -f "$SRC_ROOT/$name/${name}.sqlite" ]]; then
    cat >> "$out_yaml" <<EOF
    - nsys_sqlite
EOF
  fi
}

copy_one() {
  local name=$1
  local desc=$2
  local hf_url=$3
  local family=$4
  local moe=$5
  local batch=$6
  local tp=$7
  local ep=$8
  local num_params=$9
  local num_emb=${10}
  local layers=${11}
  local heads=${12}
  local head_dim=${13}
  local active_params=${14}

  local src="$SRC_ROOT/$name"
  local dst="$BUNDLE_ROOT/$name"
  if [[ ! -d "$src" ]]; then
    echo "skip missing $src" >&2
    return 0
  fi

  rm -rf "$dst"
  mkdir -p "$dst/logs"

  cp "$src/bench_results.jsonl" "$dst/bench_results.jsonl"
  [[ -f "$src/${name}.sqlite" ]] && cp "$src/${name}.sqlite" "$dst/${name}.sqlite"
  [[ -f "$src/${name}.nsys-rep" ]] && cp "$src/${name}.nsys-rep" "$dst/${name}.nsys-rep"
  cp "$src/${name}.server.log" "$dst/logs/${name}.server.log"
  cp "$src/${name}.client.log" "$dst/logs/${name}.client.log"

  write_yaml "$dst/${name}.yaml" "$name" "$desc" "$hf_url" "$family" "$moe" "$batch" "$tp" "$ep" \
    "$num_params" "$num_emb" "$layers" "$heads" "$head_dim" "$active_params"

  cat > "$dst/README.md" <<EOF
# ${name}

SGLang online serving benchmark on one Perlmutter A100 node.

- Input/output tokens: ${INPUT_LEN}/${OUTPUT_LEN}
- Batch/concurrency: ${batch}
- Framework: SGLang ${SGLANG_VERSION}
- Communication library: ${COMM_LIBRARY_NAME}
- Server trace: NSYS SQLite + NSYS report
- Client metrics: \`bench_results.jsonl\` from \`sglang.bench_serving\`

The server was launched with \`--disable-cuda-graph\`,
\`--disable-piecewise-cuda-graph\`, and \`--disable-custom-all-reduce\` so
tensor-parallel collectives route through the selected distributed backend.

The client used \`sglang.bench_serving --dataset-name random-ids
--tokenize-prompt\` so the prompt length is exactly controlled without relying
on ShareGPT download/tokenization.
EOF

  echo "created $dst"
}

copy_one "qwen3-4b-sglang-tp4-batch8-perlmutter" \
  "Qwen3-4B SGLang inference, TP=4, batch size 8." \
  "https://huggingface.co/Qwen/Qwen3-4B" "Qwen3-4B" "false" 8 4 1 \
  4022468096 388956160 36 32 128 0

copy_one "qwen3-4b-sglang-tp4-batch128-perlmutter" \
  "Qwen3-4B SGLang inference, TP=4, batch size 128." \
  "https://huggingface.co/Qwen/Qwen3-4B" "Qwen3-4B" "false" 128 4 1 \
  4022468096 388956160 36 32 128 0

copy_one "llama-3.1-8b-sglang-tp4-batch8-perlmutter" \
  "Llama-3.1-8B SGLang inference, TP=4, batch size 8." \
  "https://huggingface.co/meta-llama/Llama-3.1-8B" "Llama-3.1-8B" "false" 8 4 1 \
  8030261248 1050673152 32 32 128 0

copy_one "llama-3.1-8b-sglang-tp4-batch128-perlmutter" \
  "Llama-3.1-8B SGLang inference, TP=4, batch size 128." \
  "https://huggingface.co/meta-llama/Llama-3.1-8B" "Llama-3.1-8B" "false" 128 4 1 \
  8030261248 1050673152 32 32 128 0

copy_one "deepseek-moe-16b-sglang-tp4-ep4-batch8-perlmutter" \
  "DeepSeek-MoE-16B SGLang inference, TP=4, EP=4, batch size 8." \
  "https://huggingface.co/deepseek-ai/deepseek-moe-16b-base" "DeepSeek-MoE-16B" "true" 8 4 4 \
  16375728128 419430400 28 16 128 2828650496

copy_one "deepseek-moe-16b-sglang-tp4-ep4-batch128-perlmutter" \
  "DeepSeek-MoE-16B SGLang inference, TP=4, EP=4, batch size 128." \
  "https://huggingface.co/deepseek-ai/deepseek-moe-16b-base" "DeepSeek-MoE-16B" "true" 128 4 4 \
  16375728128 419430400 28 16 128 2828650496

echo
echo "Bundle root: $BUNDLE_ROOT"
find "$BUNDLE_ROOT" -maxdepth 2 -type f | sort
