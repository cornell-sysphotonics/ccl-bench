#!/usr/bin/env bash
set -euo pipefail

# Minimal SGLang request smoke for Perlmutter.
# This bypasses sglang.bench_serving and sends one /generate request directly,
# so we can distinguish server/request-path failures from benchmark-client bugs.

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-minimal-debug}"
MODEL_ID="${MODEL_ID:-$PSCRATCH/models/Qwen3-4B}"
TP="${TP:-1}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
CONTEXT_LEN="${CONTEXT_LEN:-1280}"
PORT="${PORT:-31991}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

mkdir -p "$TRACE_ROOT"

export HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# Keep all runtime/JIT caches off $HOME. SGLang/FlashInfer/Triton may take
# locks or compile kernels on first request.
JOB_CACHE_BASE="${JOB_CACHE_BASE:-/tmp/sglang_${SLURM_JOB_ID:-manual}_$$}"
mkdir -p "$JOB_CACHE_BASE"
export XDG_CACHE_HOME="$JOB_CACHE_BASE/xdg"
export FLASHINFER_WORKSPACE_BASE="$JOB_CACHE_BASE/flashinfer"
export TVM_FFI_CACHE_DIR="$JOB_CACHE_BASE/tvm-ffi"
export TRITON_CACHE_DIR="$JOB_CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_BASE/torchinductor"
export CUDA_CACHE_PATH="$JOB_CACHE_BASE/cuda"
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TVM_FFI_CACHE_DIR"

server_log="$TRACE_ROOT/server_tp${TP}.log"
client_log="$TRACE_ROOT/client_tp${TP}.log"
response_json="$TRACE_ROOT/response_tp${TP}.json"

echo "Host: $(hostname)"
echo "Python: $(command -v python3)"
python3 - <<'PY'
import torch, sglang
print(f"Torch: {torch.__version__} CUDA={torch.version.cuda}")
print(f"SGLang: {getattr(sglang, '__version__', 'unknown')}")
PY
echo "TRACE_ROOT=$TRACE_ROOT"
echo "JOB_CACHE_BASE=$JOB_CACHE_BASE"
echo "MODEL_ID=$MODEL_ID TP=$TP INPUT_LEN=$INPUT_LEN OUTPUT_LEN=$OUTPUT_LEN"

server_args=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_ID"
  --host 127.0.0.1
  --port "$PORT"
  --tp-size "$TP"
  --context-length "$CONTEXT_LEN"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --trust-remote-code
  --skip-server-warmup
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
  --log-level info
)

echo "Starting server:"
printf '  %q' "${server_args[@]}"
echo
"${server_args[@]}" > "$server_log" 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" >/dev/null 2>&1; then
    kill -INT "$server_pid" >/dev/null 2>&1 || true
    sleep 5
    kill "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start=$(date +%s)
while true; do
  if curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    echo "server exited before ready" >&2
    tail -120 "$server_log" >&2 || true
    exit 1
  fi
  if (( $(date +%s) - start > TIMEOUT_SECONDS )); then
    echo "server readiness timeout" >&2
    tail -160 "$server_log" >&2 || true
    exit 1
  fi
  sleep 2
done
echo "server ready"

python3 - "$PORT" "$INPUT_LEN" "$OUTPUT_LEN" "$response_json" > "$client_log" 2>&1 <<'PY'
import json
import sys
import time
import requests

port = int(sys.argv[1])
input_len = int(sys.argv[2])
output_len = int(sys.argv[3])
out_path = sys.argv[4]

payload = {
    "input_ids": [100] * input_len,
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": output_len,
        "ignore_eos": True,
    },
    "stream": False,
}

url = f"http://127.0.0.1:{port}/generate"
print("POST", url, "input_len", input_len, "output_len", output_len, flush=True)
start = time.perf_counter()
resp = requests.post(url, json=payload, timeout=300)
elapsed = time.perf_counter() - start
print("status", resp.status_code, "elapsed_s", elapsed, flush=True)
print(resp.text[:2000], flush=True)
resp.raise_for_status()
data = resp.json()
with open(out_path, "w") as f:
    json.dump({"elapsed_s": elapsed, "response": data}, f, indent=2)
PY

echo "client done"
tail -80 "$client_log"
echo "response: $response_json"
