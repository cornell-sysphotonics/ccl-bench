#!/bin/bash
# Run script for deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-mscclpp-perlmutter
# Framework: TorchTitan  |  Model: deepseek-v3-16b  |  TP=4  DP_shard=2  PP=1  EP=8
set -euo pipefail

NUM_NODES=2
GPUS_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

# Activate torchtitan environment and run
torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  torchtitan/train.py \
  --job.config_file trace_collection/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-mscclpp-perlmutter/workload.toml \
  --parallelism.tensor_parallel_degree 4 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.expert_parallel_degree 8 \
  --profiling.enable_profiling true \
  --profiling.save_traces_folder ./profile_traces
