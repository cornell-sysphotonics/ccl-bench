#!/bin/bash
# Run script for deepseek-moe-16b-sglang-tp4-ep4-batch8-puremscclpp-perlmutter
# Framework: SGLang  |  Model: DeepSeek-MoE-16B  |  TP=4  EP=4
set -euo pipefail

MODEL="deepseek-ai/deepseek-moe-16b-base"
TP=4
EP=4
BATCH_SIZE=8
INPUT_LEN=1024
OUTPUT_LEN=128
PORT=30000

export NCCL_IB_DISABLE="1"

# Launch SGLang server with Nsight Systems profiling
nsys profile \
  --output=deepseek-moe-16b-sglang-tp4-ep4-batch8-puremscclpp-perlmutter_nsys \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --force-overwrite=true \
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 4 \
    --ep 4 \
    --port $PORT &

SERVER_PID=$!

echo "Waiting for SGLang server..."
for i in $(seq 1 120); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
done

# Run benchmark
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://localhost:$PORT \
  --num-prompts $BATCH_SIZE \
  --input-len $INPUT_LEN \
  --output-len $OUTPUT_LEN

kill $SERVER_PID || true
