#!/bin/bash
#
# Experiment: Qwen3-32B, TP=4, Batch=4, Burst Load (SGLang)
#
# This script runs the NVLink profiling experiment with specific configuration.
# Assumes nsys_profile_server.sh and other required files are in this directory.
#

set -e

# Experiment configuration
export MODEL="Qwen/Qwen3-32B"
export TP_SIZE=4
export MAX_NUM_SEQS=4
export MAX_BATCHED_TOKENS=$((MAX_NUM_SEQS * 7315))
export NUM_PROMPTS=$((MAX_NUM_SEQS * 3))
export REQUEST_RATE=inf

# Dataset configuration for bursty load
export DATASET_NAME="burstgpt"
export DATASET_PATH="BurstGPT_without_fails_1.csv"
export USE_TIMESTAMPS="true"
export TIME_SCALE=0.5

# Output to PSCRATCH directory
EXPERIMENT_NAME="qwen3-32b-tp_4-batch_4-burst-sglang-group_11"
export OUTPUT_DIR="$PSCRATCH/nvlink_experiments/$EXPERIMENT_NAME"
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "========================================"
echo "Experiment: qwen3-32b-tp_4-batch_4-burst (SGLang)"
echo "========================================"
echo "Model: $MODEL"
echo "Tensor Parallel: $TP_SIZE"
echo "Batch Size: $MAX_NUM_SEQS"
echo "Max Batched Tokens: $MAX_BATCHED_TOKENS"
echo "Num Prompts: $NUM_PROMPTS"
echo "Dataset: $DATASET_NAME ($DATASET_PATH)"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Backend: SGLang"
echo "========================================"
echo ""

# Run the profiling script
./nsys_profile_server.sh

# Save experiment metadata
cat > "$OUTPUT_DIR/experiment_metadata.json" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "model": "$MODEL",
  "backend": "sglang",
  "tensor_parallel_size": $TP_SIZE,
  "batch_size": $MAX_NUM_SEQS,
  "max_batched_tokens": $MAX_BATCHED_TOKENS,
  "num_prompts": $NUM_PROMPTS,
  "request_rate": "$REQUEST_RATE",
  "dataset_name": "$DATASET_NAME",
  "dataset_path": "$DATASET_PATH",
  "use_timestamps": "$USE_TIMESTAMPS",
  "time_scale": $TIME_SCALE,
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo ""
echo "========================================"
echo "Experiment Complete!"
echo "========================================"
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/"
echo ""

