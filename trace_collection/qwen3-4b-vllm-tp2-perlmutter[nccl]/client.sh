#!/usr/bin/env bash
# Exact client command for variant: qwen3-4b-vllm-tp2-perlmutter[nccl]
# Run in a second shell on the same compute node, after server.sh is ready.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_PATH="${MODEL_PATH:-$PSCRATCH/models/Qwen3-4B}"

python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 8
