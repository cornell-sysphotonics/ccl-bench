#!/bin/bash
# Run script for llama-3.1-8b-vllm-tp4-batch128-perlmutter
# Framework: vLLM  |  Model: llama-3.1-8b  |  TP=4  EP=1
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B"
TP=4
EP=1
BATCH_SIZE=128
INPUT_LEN=1024
OUTPUT_LEN=128
NUM_GPUS=4
PORT=8000

export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch vLLM server with Nsight Systems profiling
nsys profile \
  --output=llama-3.1-8b-vllm-tp4-batch128-perlmutter_nsys \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --force-overwrite=true \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --enable-expert-parallel \
    --max-num-seqs $BATCH_SIZE \
    --port $PORT &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
done

# Run benchmark
python -m vllm.entrypoints.benchmark_serving \
  --model "$MODEL" \
  --backend vllm \
  --endpoint /v1/completions \
  --num-prompts $BATCH_SIZE \
  --input-len $INPUT_LEN \
  --output-len $OUTPUT_LEN

kill $SERVER_PID || true
