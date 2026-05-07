#!/bin/bash
# Run script for llama-3.1-8b-sglang-tp4-batch8-puremscclpp-perlmutter
# Framework: SGLang  |  Model: Llama-3.1-8B  |  TP=4  EP=1
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B"
TP=4
EP=1
BATCH_SIZE=8
INPUT_LEN=1024
OUTPUT_LEN=128
PORT=30000

export NCCL_IB_DISABLE="1"

# Launch SGLang server with Nsight Systems profiling
nsys profile \
  --output=llama-3.1-8b-sglang-tp4-batch8-puremscclpp-perlmutter_nsys \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --force-overwrite=true \
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 4 \
    --ep 1 \
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
