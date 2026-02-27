#!/bin/bash

# Example usage:
# ./scripts/get_comm_kernel_breakdown_tpu_group4.sh <TRACE_DIR>
# e.g. ./scripts/get_comm_kernel_breakdown_tpu_group4.sh ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4

if [ -z "$1" ]; then
  echo "Usage: $0 <TRACE_DIR>"
  exit 1
fi

TRACE_DIR=$1

python3 ./tools/main.py --trace "$TRACE_DIR" --metric "comm_kernel_breakdown_tpu"
