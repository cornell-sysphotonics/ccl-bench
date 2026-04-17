#!/usr/bin/env bash
# Exact server command for variant: qwen3-4b-vllm-tp4-perlmutter[mscclpp]
# Run on a Perlmutter compute node (4xA100), inside cvllm conda env.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$PSCRATCH/models/Qwen3-4B}"
PORT="${PORT:-8000}"
PREFIX="${PREFIX:-$PSCRATCH/ccl-bench-traces/qwen3-4b-perlmutter/qwen3-4b-tp4-perlmutter[mscclpp]-main_$(date +%Y%m%d_%H%M%S)}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export LD_PRELOAD=$HOME/mscclpp/build/lib/libmscclpp_nccl.so

nsys profile \
  -t cuda,nvtx,osrt -s none --cpuctxsw=none \
  --trace-fork-before-exec=true --force-overwrite=true \
  --stats=true --export=sqlite \
  -o "$PREFIX" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --enforce-eager \
    --tensor-parallel-size 4 \
    --port "$PORT"
