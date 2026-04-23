#!/usr/bin/env bash
set -uo pipefail

# SGLang server+client inference collection on Perlmutter.
#
# Collects one NSYS SQLite trace plus one SGLang bench_serving JSONL per row.
# Defaults are chosen to match the single-node 4xA100 vLLM rows:
#   input_len=1024, output_len=128, TP=4, batches 8 and 128.
#
# This script expects to run inside a Slurm allocation with 4 GPUs.
# It also expects the SGLang conda env to be active. On Perlmutter we use:
#   /pscratch/sd/k/kg597/session1/csglang

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-inference}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
CONTEXT_MARGIN="${CONTEXT_MARGIN:-128}"
PORT_BASE="${PORT_BASE:-31080}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-900}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"
VARIANTS="${VARIANTS:-qwen_b8 qwen_b128 llama_b8 llama_b128 deepseek_b8 deepseek_b128}"
PROFILE_NSYS="${PROFILE_NSYS:-0}"
NSYS_TRACE_TYPES="${NSYS_TRACE_TYPES:-cuda,nvtx}"
NSYS_STOP_WAIT_SECONDS="${NSYS_STOP_WAIT_SECONDS:-420}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
CLIENT_DISABLE_STREAM="${CLIENT_DISABLE_STREAM:-1}"
CLIENT_WARMUP_REQUESTS="${CLIENT_WARMUP_REQUESTS:-0}"
DATASET_NAME="${DATASET_NAME:-random-ids}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-auto}"
DISABLE_FLASHINFER_AUTOTUNE="${DISABLE_FLASHINFER_AUTOTUNE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"

COMM_BACKEND="${COMM_BACKEND:-nccl}"
MSCCLPP_PRELOAD="${MSCCLPP_PRELOAD:-$HOME/mscclpp/build/lib/libmscclpp_nccl.so}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# Workaround for Perlmutter: home directories do not support flock(), and both
# FlashInfer and TVM FFI/SGL kernel JIT take file locks on first request. Keep
# all runtime/JIT caches on node-local /tmp.
JOB_CACHE_BASE="${JOB_CACHE_BASE:-/tmp/sglang_${SLURM_JOB_ID:-manual}_$$}"
mkdir -p "$JOB_CACHE_BASE"
export XDG_CACHE_HOME="$JOB_CACHE_BASE/xdg"
export FLASHINFER_WORKSPACE_BASE="$JOB_CACHE_BASE/flashinfer"
export TVM_FFI_CACHE_DIR="$JOB_CACHE_BASE/tvm-ffi"
export TRITON_CACHE_DIR="$JOB_CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_BASE/torchinductor"
export CUDA_CACHE_PATH="$JOB_CACHE_BASE/cuda"
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" "$TVM_FFI_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

if [[ "$COMM_BACKEND" == "gloo" ]]; then
  export CCL_BENCH_DIST_BACKEND=gloo
  export CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  DISABLE_CUSTOM_ALL_REDUCE=1
elif [[ "$COMM_BACKEND" == "mscclpp" ]]; then
  export LD_PRELOAD="$MSCCLPP_PRELOAD"
  # MSCCL++ rejects expandable-segments on A100
  unset PYTORCH_CUDA_ALLOC_CONF
elif [[ "$COMM_BACKEND" == "pure_mscclpp" ]]; then
  export LD_PRELOAD="$MSCCLPP_PRELOAD"
  export VLLM_NCCL_SO_PATH="$MSCCLPP_PRELOAD"
  export SGLANG_NCCL_SO_PATH="$MSCCLPP_PRELOAD"
  unset MSCCLPP_NCCL_LIB_PATH
  unset PYTORCH_CUDA_ALLOC_CONF
  DISABLE_CUSTOM_ALL_REDUCE=1
elif [[ "$COMM_BACKEND" == "native_mscclpp" ]]; then
  # Use SGLang's built-in MSCCL++ support without LD_PRELOAD
  DISABLE_CUSTOM_ALL_REDUCE=0
fi

mkdir -p "$TRACE_ROOT"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

wait_for_server() {
  local port=$1
  local log=$2
  local server_pid=$3
  local start
  start=$(date +%s)
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/model_info" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "server process exited before becoming healthy: $log" >&2
      tail -80 "$log" >&2
      return 1
    fi
    if (( $(date +%s) - start > SERVER_WAIT_SECONDS )); then
      echo "timed out waiting for SGLang server on port $port" >&2
      tail -120 "$log" >&2
      return 1
    fi
    sleep 5
  done
}

stop_server() {
  local pid=$1
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -INT "$pid" >/dev/null 2>&1 || true
    for _ in $(seq 1 "$NSYS_STOP_WAIT_SECONDS"); do
      kill -0 "$pid" >/dev/null 2>&1 || return 0
      sleep 1
    done
    kill "$pid" >/dev/null 2>&1 || true
  fi
}

variant_config() {
  case "$1" in
    qwen_b8)
      echo "qwen3-4b-sglang-tp4-batch8-perlmutter|/pscratch/sd/k/kg597/models/Qwen3-4B|4|1|8|Qwen3-4B|false"
      ;;
    qwen_b128)
      echo "qwen3-4b-sglang-tp4-batch128-perlmutter|/pscratch/sd/k/kg597/models/Qwen3-4B|4|1|128|Qwen3-4B|false"
      ;;
    llama_b8)
      echo "llama-3.1-8b-sglang-tp4-batch8-perlmutter|meta-llama/Llama-3.1-8B|4|1|8|Llama-3.1-8B|false"
      ;;
    llama_b128)
      echo "llama-3.1-8b-sglang-tp4-batch128-perlmutter|meta-llama/Llama-3.1-8B|4|1|128|Llama-3.1-8B|false"
      ;;
    deepseek_b8)
      echo "deepseek-moe-16b-sglang-tp4-ep4-batch8-perlmutter|deepseek-ai/deepseek-moe-16b-base|4|4|8|DeepSeek-MoE-16B|true"
      ;;
    deepseek_b128)
      echo "deepseek-moe-16b-sglang-tp4-ep4-batch128-perlmutter|deepseek-ai/deepseek-moe-16b-base|4|4|128|DeepSeek-MoE-16B|true"
      ;;
    *)
      echo "unknown variant: $1" >&2
      return 1
      ;;
  esac
}

run_one() {
  local key=$1
  local cfg name model tp ep batch family moe out_dir port server_log client_log nsys_prefix results_jsonl server_pid

  cfg=$(variant_config "$key") || return 1
  IFS="|" read -r name model tp ep batch family moe <<< "$cfg"

  out_dir="$TRACE_ROOT/$name"
  port=$((PORT_BASE + RANDOM % 1000))
  server_log="$out_dir/${name}.server.log"
  client_log="$out_dir/${name}.client.log"
  nsys_prefix="$out_dir/${name}"
  results_jsonl="$out_dir/bench_results.jsonl"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  echo
  echo "============================================================"
  echo "RUN $name  (SGLang server+bench_serving)"
  echo "  model: $model"
  echo "  TP=$tp EP=$ep batch=$batch input=$INPUT_LEN output=$OUTPUT_LEN"
  echo "  out: $out_dir"
  echo "============================================================"

  local server_args=(
    python3 -m sglang.launch_server
    --model-path "$model"
    --host 127.0.0.1
    --port "$port"
    --tp-size "$tp"
    --context-length "$((INPUT_LEN + OUTPUT_LEN + CONTEXT_MARGIN))"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG_TIMEOUT"
    --log-level info
  )

  if [[ "$DISABLE_FLASHINFER_AUTOTUNE" == "1" ]]; then
    server_args+=(--disable-flashinfer-autotune)
  fi
  if [[ -n "$ATTENTION_BACKEND" ]]; then
    server_args+=(--attention-backend "$ATTENTION_BACKEND")
  fi
  if [[ -n "$SAMPLING_BACKEND" ]]; then
    server_args+=(--sampling-backend "$SAMPLING_BACKEND")
  fi
  if [[ "$SKIP_SERVER_WARMUP" == "1" ]]; then
    server_args+=(--skip-server-warmup)
  fi
  if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
    server_args+=(--disable-cuda-graph)
  elif [[ "$CUDA_GRAPH_BS" == "auto" ]]; then
    server_args+=(--cuda-graph-bs "$batch")
  elif [[ -n "$CUDA_GRAPH_BS" ]]; then
    server_args+=(--cuda-graph-bs $CUDA_GRAPH_BS)
  fi
  if [[ "$DISABLE_PIECEWISE_CUDA_GRAPH" == "1" ]]; then
    server_args+=(--disable-piecewise-cuda-graph)
  fi
  if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
    server_args+=(--disable-custom-all-reduce)
  fi
  if [[ "$COMM_BACKEND" == "native_mscclpp" || "$COMM_BACKEND" == "mscclpp" ]]; then
    server_args+=(--enable-mscclpp)
  fi

  if [[ "$ep" != "1" ]]; then
    server_args+=(--ep-size "$ep" --moe-dense-tp-size 1)
  fi

  if [[ "$PROFILE_NSYS" == "1" ]]; then
    nsys profile \
      -t "$NSYS_TRACE_TYPES" \
      -s none --cpuctxsw=none \
      --trace-fork-before-exec=true \
      --force-overwrite=true \
      --stats=true \
      --export=sqlite \
      -o "$nsys_prefix" \
      "${server_args[@]}" \
      > "$server_log" 2>&1 &
  else
    "${server_args[@]}" > "$server_log" 2>&1 &
  fi
  server_pid=$!

  if ! wait_for_server "$port" "$server_log" "$server_pid"; then
    stop_server "$server_pid"
    wait "$server_pid" >/dev/null 2>&1 || true
    return 1
  fi

  echo "  server ready on port $port"

  local client_args=(
    python3 -m sglang.bench_serving
    --backend sglang \
    --host 127.0.0.1 \
    --port "$port" \
    --dataset-name "$DATASET_NAME" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 1.0 \
    --num-prompts "$batch" \
    --max-concurrency "$batch" \
    --model "$model" \
    --tokenizer "$model" \
    --output-file "$results_jsonl" \
    --output-details \
    --disable-tqdm \
    --warmup-requests "$CLIENT_WARMUP_REQUESTS"
  )
  if [[ "$DATASET_NAME" == "random-ids" ]]; then
    client_args+=(--tokenize-prompt)
  fi
  if [[ "$CLIENT_DISABLE_STREAM" == "1" ]]; then
    client_args+=(--disable-stream)
  fi

  "${client_args[@]}" > "$client_log" 2>&1
  local client_rc=$?

  stop_server "$server_pid"
  wait "$server_pid" >/dev/null 2>&1 || true

  if [[ "$client_rc" != 0 ]]; then
    echo "  ! bench_serving failed rc=$client_rc; see $client_log" >&2
    return "$client_rc"
  fi

  if [[ ! -s "$results_jsonl" ]]; then
    echo "  ! missing bench_results.jsonl: $results_jsonl" >&2
    return 1
  fi
  if [[ "$PROFILE_NSYS" == "1" && ! -s "${nsys_prefix}.sqlite" ]]; then
    echo "  ! missing NSYS sqlite: ${nsys_prefix}.sqlite" >&2
    return 1
  fi

  echo "  produced:"
  find "$out_dir" -maxdepth 1 -type f | sort | sed "s#^#    #"
}

main() {
  if [[ "$PROFILE_NSYS" == "1" ]]; then
    require_cmd nsys
  fi
  require_cmd curl
  require_cmd nvidia-smi

  echo "Host: $(hostname)"
  echo "GPU count: $(nvidia-smi -L | wc -l)"
  echo "Trace root: $TRACE_ROOT"
  echo "Python: $(command -v python3)"
  python3 - <<'PY'
import torch, sglang
print(f"Torch: {torch.__version__} CUDA={torch.version.cuda}")
print(f"SGLang: {getattr(sglang, '__version__', 'unknown')}")
PY
  echo "Variants: $VARIANTS"
  echo "Profile NSYS: $PROFILE_NSYS"

  local failures=0
  for key in $VARIANTS; do
    run_one "$key" || failures=$((failures + 1))
  done

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sort | sed "s#^#  #"
  return "$failures"
}

main "$@"
