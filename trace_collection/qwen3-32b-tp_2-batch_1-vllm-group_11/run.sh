#!/bin/bash
#
# Experiment: Qwen3-32B, TP=2, Batch=1
#
# This script runs the NVLink profiling experiment with specific configuration.
# Assumes nsys_profile_server.sh and other required files are in this directory.
#

set -e

# Experiment configuration
export MODEL="Qwen/Qwen3-32B"
export TP_SIZE=2
export MAX_NUM_SEQS=1
export MAX_BATCHED_TOKENS=$((MAX_NUM_SEQS * 2048))
export NUM_PROMPTS=$((MAX_NUM_SEQS * 3))
export REQUEST_RATE=inf

# Output to PSCRATCH directory
EXPERIMENT_NAME="qwen3-32b-tp_2-batch_1-vllm-group_11"
export OUTPUT_DIR="$PSCRATCH/nvlink_experiments/$EXPERIMENT_NAME"
export CUDA_VISIBLE_DEVICES=0,1

echo "========================================"
echo "Experiment: qwen3-32b-tp_2-batch_1"
echo "========================================"
echo "Model: $MODEL"
echo "Tensor Parallel: $TP_SIZE"
echo "Batch Size: $MAX_NUM_SEQS"
echo "Max Batched Tokens: $MAX_BATCHED_TOKENS"
echo "Num Prompts: $NUM_PROMPTS"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================"
echo ""

# Run the profiling script
./nsys_profile_server.sh

# Save experiment metadata
cat > "$OUTPUT_DIR/experiment_metadata.json" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "model": "$MODEL",
  "tensor_parallel_size": $TP_SIZE,
  "batch_size": $MAX_NUM_SEQS,
  "max_batched_tokens": $MAX_BATCHED_TOKENS,
  "num_prompts": $NUM_PROMPTS,
  "request_rate": "$REQUEST_RATE",
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
