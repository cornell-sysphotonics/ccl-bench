#!/bin/bash
# Run script for nanochat-8b-pytorch-dp-2-dgx_spark
# Framework: PyTorch
set -euo pipefail

NUM_GPUS=2

torchrun \
  --nproc_per_node=$NUM_GPUS \
  train.py
