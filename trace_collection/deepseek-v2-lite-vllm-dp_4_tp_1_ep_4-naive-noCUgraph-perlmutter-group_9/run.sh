#!/bin/bash
# Run script for deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-naive-noCUgraph-perlmutter-group_9
# Framework: vLLM  |  Model: deepseek_v2  |  TP=1  EP=4
set -euo pipefail

MODEL="deepseek-ai/DeepSeek-V2-Lite/tree/main"
TP=1
EP=4
BATCH_SIZE=1
INPUT_LEN=1024
OUTPUT_LEN=128
NUM_GPUS=4
PORT=8000

export NCCL_IB_QPS_PER_CONNECTION="None"
export VLLM_ALL2ALL_BACKEND="all-reduce"

# Launch vLLM server with Nsight Systems profiling
nsys profile \
  --output=deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-naive-noCUgraph-perlmutter-group_9_nsys \
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
