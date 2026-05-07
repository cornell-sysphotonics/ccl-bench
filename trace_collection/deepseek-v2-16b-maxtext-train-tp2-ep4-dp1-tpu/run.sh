#!/bin/bash
# Run script for deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu
# Framework: MaxText  |  Model: deepseek-v2-16b  |  TP=2  EP=4
set -euo pipefail

MAXTEXT_ROOT="${MAXTEXT_ROOT:-./MaxText}"
MODEL_NAME="deepseek-v2-16b"
TP=2
EP=4
PER_DEVICE_BATCH=8
SEQ_LEN=1024
NUM_DEVICES=8

python $MAXTEXT_ROOT/train.py \
  MaxText/configs/base.yml \
  model_name=$MODEL_NAME \
  ici_mesh_shape="[1, $TP, 1, $EP]" \
  per_device_batch_size=$PER_DEVICE_BATCH \
  max_target_length=$SEQ_LEN \
  steps=10 \
  enable_profiler=true \
  profile_dir=./profiles
