#!/bin/bash
# Run script for Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_64_16384]-group12
# Framework: vLLM  |  Model: mixtral  |  TP=1  EP=4
set -euo pipefail

MODEL="mistralai/Mixtral-8x7B-v0.1/tree/main"
TP=1
EP=4
BATCH_SIZE=64
INPUT_LEN=1024
OUTPUT_LEN=128
NUM_GPUS=4
PORT=8000

export NCCL_IB_QPS_PER_CONNECTION=""

# Launch vLLM server with Nsight Systems profiling
nsys profile \
  --output=Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_64_16384]-group12_nsys \
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
