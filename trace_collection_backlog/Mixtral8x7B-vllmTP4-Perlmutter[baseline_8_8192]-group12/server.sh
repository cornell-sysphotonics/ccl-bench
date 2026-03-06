#!/bin/bash

MODEL=${1:-"mistralai/Mixtral-8x7B-v0.1"}
PORT=${2:-8000}

# ===== baseline defaults =====
BATCH_TOKENS=${3:-8192}     # max-num-batched-tokens
MAX_SEQS=${4:-8}            # max-num-seqs
MAX_MODEL_LEN=${5:-4096}    # max-model-len
# =============================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"

LOG_FILE="${LOGS_DIR}/vllm_baseline_batch${BATCH_TOKENS}_seq${MAX_SEQS}_modellen${MAX_MODEL_LEN}.log"
NSYS_OUTPUT="${LOGS_DIR}/vllm_baseline_profile_batch${BATCH_TOKENS}_seq${MAX_SEQS}_modellen${MAX_MODEL_LEN}"

# HuggingFace cache
HF_CACHE_DIR="${SCRIPT_DIR}/../../hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

echo "========================================="
echo "Launching vLLM BASELINE (default Triton MoE)"
echo "Model:          $MODEL"
echo "Port:           $PORT"
echo "Batch Tokens:   $BATCH_TOKENS"
echo "Max Seqs:       $MAX_SEQS"
echo "Max Model Len:  $MAX_MODEL_LEN"
echo "========================================="

export VLLM_LOGGING_LEVEL=INFO

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output="$NSYS_OUTPUT" \
  --force-overwrite=true \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$BATCH_TOKENS" \
    --max-num-seqs "$MAX_SEQS" \
    --enforce-eager \
    --disable-log-requests \
  2>&1 | tee "$LOG_FILE"

rm -rf "$TORCH_EXTENSIONS_DIR"