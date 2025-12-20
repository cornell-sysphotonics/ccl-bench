#!/bin/bash

# Example usage:
# ./scripts/get_comm_kernel_breakdown_tpu.sh <TRACE_DIR>
# e.g. ./scripts/get_comm_kernel_breakdown_tpu.sh ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4

TRACE_DIR=${1:-"./trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4"}

python ./tools/main.py --trace "$TRACE_DIR" --metric "comm_kernel_breakdown_tpu"
