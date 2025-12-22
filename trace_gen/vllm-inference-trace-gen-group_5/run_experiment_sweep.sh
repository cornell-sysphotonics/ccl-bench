#!/bin/bash
set -euo pipefail

BASE_TRACE_ROOT="/pscratch/sd/b/bck/inference_sweep_output"
BENCH_NUM_PROMPTS=50
BENCH_QPS=10

# Uncomment whichever experiments you would like to run!
EXPERIMENTS=(
    # Llama 3 Pure Parallelism
    # "llama3_2gpu_dp2:model=meta-llama/Llama-3.1-8B;dp=2;tp=1;pp=1;gpus=0,1"
    # "llama3_2gpu_tp2:model=meta-llama/Llama-3.1-8B;dp=1;tp=2;pp=1;gpus=0,1"
    # "llama3_2gpu_pp2:model=meta-llama/Llama-3.1-8B;dp=1;tp=1;pp=2;gpus=0,1"
    # "llama3_4gpu_dp4:model=meta-llama/Llama-3.1-8B;dp=4;tp=1;pp=1;gpus=0,1,2,3"
    # "llama3_4gpu_tp4:model=meta-llama/Llama-3.1-8B;dp=1;tp=4;pp=1;gpus=0,1,2,3"
    # "llama3_4gpu_pp4:model=meta-llama/Llama-3.1-8B;dp=1;tp=1;pp=4;gpus=0,1,2,3"

    # Qwen 3 Pure Parallelism
    # "qwen3_2gpu_dp2:model=Qwen/Qwen3-8B;dp=2;tp=1;pp=1;gpus=0,1"
    # "qwen3_2gpu_tp2:model=Qwen/Qwen3-8B;dp=1;tp=2;pp=1;gpus=0,1"
    # "qwen3_2gpu_pp2:model=Qwen/Qwen3-8B;dp=1;tp=1;pp=2;gpus=0,1"
    # "qwen3_4gpu_dp4:model=Qwen/Qwen3-8B;dp=4;tp=1;pp=1;gpus=0,1,2,3"
    # "qwen3_4gpu_tp4:model=Qwen/Qwen3-8B;dp=1;tp=4;pp=1;gpus=0,1,2,3"
    # "qwen3_4gpu_pp4:model=Qwen/Qwen3-8B;dp=1;tp=1;pp=4;gpus=0,1,2,3"

    # Mistral 7B Pure Parallelism
    # "mistral_2gpu_dp2:model=mistralai/Mistral-7B-Instruct-v0.2;dp=2;tp=1;pp=1;gpus=0,1"
    # "mistral_2gpu_tp2:model=mistralai/Mistral-7B-Instruct-v0.2;dp=1;tp=2;pp=1;gpus=0,1"
    # "mistral_2gpu_pp2:model=mistralai/Mistral-7B-Instruct-v0.2;dp=1;tp=1;pp=2;gpus=0,1"
    # "mistral_4gpu_tp4:model=mistralai/Mistral-7B-Instruct-v0.2;dp=1;tp=4;pp=1;gpus=0,1,2,3"
    # "mistral_4gpu_pp4:model=mistralai/Mistral-7B-Instruct-v0.2;dp=1;tp=1;pp=4;gpus=0,1,2,3"
    # "mistral_4gpu_dp4:model=mistralai/Mistral-7B-Instruct-v0.2;dp=4;tp=1;pp=1;gpus=0,1,2,3"

    # Llama 3 Mixed Parallelism
    # "llama3_4gpu_tp_2_dp2:model=meta-llama/Llama-3.1-8B;dp=2;tp=2;pp=1;gpus=0,1,2,3"
    # "llama3_4gpu_tp_2_pp2:model=meta-llama/Llama-3.1-8B;dp=1;tp=2;pp=2;gpus=0,1,2,3"
    # "llama3_4gpu_dp2_pp2:model=meta-llama/Llama-3.1-8B;dp=2;tp=1;pp=2;gpus=0,1,2,3"

    # Qwen 3 Mixed Parallelism
    # "qwen3_4gpu_tp2_dp2:model=Qwen/Qwen3-8B;dp=2;tp=2;pp=1;gpus=0,1,2,3"
    # "qwen3_4gpu_tp2_pp2:model=Qwen/Qwen3-8B;dp=1;tp=2;pp=2;gpus=0,1,2,3"
    # "qwen3_4gpu_dp2_pp2:model=Qwen/Qwen3-8B;dp=2;tp=1;pp=2;gpus=0,1,2,3"

    # Mistral 7B Mixed Parallelism
    # "mistral_4gpu_tp2_dp2:model=mistralai/Mistral-7B-Instruct-v0.2;dp=2;tp=2;pp=1;gpus=0,1,2,3"
    # "mistral_4gpu_tp2_pp2:model=mistralai/Mistral-7B-Instruct-v0.2;dp=1;tp=2;pp=2;gpus=0,1,2,3"
    # "mistral_4gpu_dp2_pp2:model=mistralai/Mistral-7B-Instruct-v0.2;dp=2;tp=1;pp=2;gpus=0,1,2,3"
)

for exp in "${EXPERIMENTS[@]}"; do
  NAME="${exp%%:*}"
  PARAMS="${exp#*:}"

  MODEL=$(echo "$PARAMS" | sed -n 's/.*model=\([^;]*\).*/\1/p')
  DP=$(echo "$PARAMS"    | sed -n 's/.*dp=\([^;]*\).*/\1/p')
  TP=$(echo "$PARAMS"    | sed -n 's/.*tp=\([^;]*\).*/\1/p')
  PP=$(echo "$PARAMS"    | sed -n 's/.*pp=\([^;]*\).*/\1/p')
  GPUS=$(echo "$PARAMS"  | sed -n 's/.*gpus=\([^;]*\).*/\1/p')

  TRACE_DIR="${BASE_TRACE_ROOT}/${NAME}"
  mkdir -p "$TRACE_DIR"

  echo "=== Running experiment: $NAME ==="
  echo "Model=$MODEL DP=$DP TP=$TP PP=$PP GPUs=$GPUS"
  echo "Trace dir: $TRACE_DIR"

  # Per-experiment profiler directory
  export VLLM_TORCH_PROFILER_DIR="$TRACE_DIR"
  export VLLM_TORCH_PROFILER_RECORD_SHAPES=0
  export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
  export VLLM_TORCH_PROFILER_WITH_STACK=0
  export VLLM_TORCH_PROFILER_WITH_FLOPS=0
  export VLLM_RPC_TIMEOUT=1800000

  CUDA_VISIBLE_DEVICES="$GPUS" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --swap-space 16 \
    --disable-log-requests \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --max-num-seqs 512 \
    --disable-sliding-window \
    --data-parallel-size "$DP" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --port 8000 &

  SERVER_PID=$!
  echo "Server PID: $SERVER_PID"
  sleep 20   # give it time to load model

  # Run benchmark client with profiling
  python3 -m vllm.entrypoints.cli.main bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 512 \
    --num-prompts "$BENCH_NUM_PROMPTS" \
    --request-rate "$BENCH_QPS" \
    --profile

  # Shut down server
  echo "Stopping server for $NAME"
  kill "$SERVER_PID" || true
  wait "$SERVER_PID" || true

  echo "Traces for $NAME are in: $TRACE_DIR"
  echo
done