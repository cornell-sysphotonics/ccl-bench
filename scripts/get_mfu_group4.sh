#!/bin/bash

# Example usage:
# ./scripts/get_mfu.sh <TRACE_DIR>
# e.g. ./scripts/get_mfu.sh ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4

# if [ -z "$1" ]; then
#   echo "Usage: $0 <TRACE_DIR>"
#   exit 1
# fi

TRACE_DIR=$1

python ./tools/main.py --trace ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp1-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp2-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp4-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Qwen3-4B-torchxla-vllm-tp1-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Qwen3-4B-torchxla-vllm-tp2-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Qwen3-4B-torchxla-vllm-tp4-tpu-group-4 --metric "mfu"
python ./tools/main.py --trace ./trace_collection/Qwen3-4B-torchxla-vllm-tp8-tpu-group-4 --metric "mfu"

