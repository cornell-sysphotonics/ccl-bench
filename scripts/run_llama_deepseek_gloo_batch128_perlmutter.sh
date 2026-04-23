#!/usr/bin/env bash
set -uo pipefail

# Llama/DeepSeek vLLM Kineto+NSYS runs with Gloo as the collective backend.
#
# These rows are apple-to-apple with the existing single-node 4xA100 TP4
# batch=128, input=1024, output=128 NCCL / NCCL+MSCCL++ / pure-MSCCL++ rows.
# Gloo is forced through a small vLLM patch under scripts/vllm_gloo_patch:
#   CCL_BENCH_DIST_BACKEND=gloo
#   CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-gloo-batch128}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1152}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS_CAP="${MAX_NUM_BATCHED_TOKENS_CAP:-16384}"
NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"

export HF_HOME
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CCL_BENCH_DIST_BACKEND=gloo
export CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1

mkdir -p "$TRACE_ROOT"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

scheduled_batched_tokens() {
  local full_tokens=$((BATCH_SIZE * (INPUT_LEN + OUTPUT_LEN)))
  if [[ "$MAX_NUM_BATCHED_TOKENS_CAP" =~ ^[0-9]+$ ]] && \
     (( MAX_NUM_BATCHED_TOKENS_CAP > 0 )) && \
     (( full_tokens > MAX_NUM_BATCHED_TOKENS_CAP )); then
    echo "$MAX_NUM_BATCHED_TOKENS_CAP"
  else
    echo "$full_tokens"
  fi
}

verify_vllm_patch() {
  local cc_path
  cc_path=$(python -c 'import vllm.distributed.device_communicators.cuda_communicator as m; print(m.__file__)' 2>/dev/null) || {
    echo "cannot import vLLM cuda_communicator; is the cvllm env active?" >&2
    return 1
  }
  if ! grep -q "CCL_BENCH_FORCE_TORCH_DISTRIBUTED" "$cc_path"; then
    echo "vLLM Gloo patch not detected at $cc_path" >&2
    echo "Run scripts/vllm_gloo_patch/install.sh first." >&2
    return 1
  fi
  echo "vLLM Gloo patch OK: $cc_path"
}

verify_gloo_in_log() {
  local log=$1
  if ! grep -q "backend=gloo" "$log"; then
    echo "no backend=gloo line in $log" >&2
    return 1
  fi
  if grep -qE "pynccl|ncclCommInit" "$log"; then
    echo "unexpected NCCL/PyNCCL initialization in $log" >&2
    grep -nE "pynccl|ncclCommInit" "$log" | head -5 >&2
    return 1
  fi
  echo "Gloo verified in $log"
}

model_args() {
  local model_key=$1
  case "$model_key" in
    llama3.1-8b)
      MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B}"
      MODEL_FAMILY="llama-3.1-8b"
      TP="4"
      EP="1"
      EXTRA_ARGS=(--tensor-parallel-size 4 --trust-remote-code)
      ;;
    deepseek-moe-16b)
      MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/deepseek-moe-16b-base}"
      MODEL_FAMILY="deepseek-moe-16b"
      TP="4"
      EP="4"
      EXTRA_ARGS=(--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code)
      ;;
    *)
      echo "unknown model key: $model_key" >&2
      return 1
      ;;
  esac
}

run_one() {
  local model_key=$1
  model_args "$model_key" || return 1

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens)

  local name="${MODEL_FAMILY}-vllm-tp${TP}"
  if [[ "$EP" != "1" ]]; then
    name="${name}-ep${EP}"
  fi
  name="${name}-batch${BATCH_SIZE}-gloo-perlmutter"

  local out_dir="$TRACE_ROOT/$name"
  local kineto_dir="$out_dir/kineto"
  local nsys_prefix="$out_dir/${name}"
  local latency_json="$out_dir/${name}.latency.json"
  local bench_log="$out_dir/${name}.bench.log"
  local profile_log="$out_dir/${name}.profile.log"

  rm -rf "$out_dir"
  mkdir -p "$kineto_dir"

  echo
  echo "============================================================"
  echo "RUN $name  (Gloo via torch.distributed)"
  echo "  model: $MODEL_ID"
  echo "  TP=$TP EP=$EP batch=$BATCH_SIZE input=$INPUT_LEN output=$OUTPUT_LEN"
  echo "  max_num_batched_tokens=$batched_tokens"
  echo "  out: $out_dir"
  echo "============================================================"

  local common_args=(
    --model "$MODEL_ID"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --batch-size "$BATCH_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$BATCH_SIZE"
    --max-num-batched-tokens "$batched_tokens"
    --enforce-eager
    --disable-detokenize
    --disable-custom-all-reduce
    "${EXTRA_ARGS[@]}"
  )

  echo "[1/2] latency run..."
  env python -m vllm.entrypoints.cli.main bench latency \
    "${common_args[@]}" \
    --num-iters-warmup "$NUM_ITERS_WARMUP" \
    --num-iters "$NUM_ITERS" \
    --output-json "$latency_json" \
    > "$bench_log" 2>&1
  local latency_rc=$?
  if [[ "$latency_rc" != 0 ]]; then
    echo "latency run failed rc=$latency_rc; see $bench_log"
    return "$latency_rc"
  fi
  verify_gloo_in_log "$bench_log" || return 1

  echo "[2/2] kineto + nsys profile run..."
  nsys profile \
    -t cuda,nvtx,osrt \
    -s none --cpuctxsw=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    -o "$nsys_prefix" \
    env python -m vllm.entrypoints.cli.main bench latency \
      "${common_args[@]}" \
      --num-iters-warmup "$PROFILE_WARMUP" \
      --num-iters 1 \
      --profile \
      --profiler-config.profiler torch \
      --profiler-config.torch_profiler_dir "$kineto_dir" \
      --profiler-config.torch_profiler_with_stack false \
      --profiler-config.torch_profiler_record_shapes true \
      --profiler-config.torch_profiler_with_memory true \
      --profiler-config.torch_profiler_with_flops true \
      --profiler-config.torch_profiler_use_gzip false \
      > "$profile_log" 2>&1
  local profile_rc=$?
  if [[ "$profile_rc" != 0 ]]; then
    echo "profile run failed rc=$profile_rc; see $profile_log"
    return "$profile_rc"
  fi
  verify_gloo_in_log "$profile_log" || return 1

  echo "produced:"
  find "$out_dir" -maxdepth 2 -type f | sed "s#^#  #"
}

main() {
  require_cmd python
  require_cmd nsys
  require_cmd nvidia-smi
  verify_vllm_patch || exit 1

  echo "Host: $(hostname)"
  echo "GPU count: $(nvidia-smi -L | wc -l)"
  echo "Trace root: $TRACE_ROOT"
  echo "Batch: $BATCH_SIZE input/output: $INPUT_LEN/$OUTPUT_LEN"

  local models
  models=(${MODELS:-llama3.1-8b deepseek-moe-16b})

  for model_key in "${models[@]}"; do
    run_one "$model_key" || echo "FAILED: $model_key"
  done

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sed "s#^#  #"
}

main "$@"
