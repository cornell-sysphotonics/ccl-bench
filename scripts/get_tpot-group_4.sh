#!/bin/bash

# Example usage:
# ./scripts/get_tpot.sh <MODEL> <TP_SIZE> [OPTIONS]
# e.g. ./scripts/get_tpot.sh meta-llama/Llama-3.1-8B-Instruct 4

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <MODEL> <TP_SIZE> [OPTIONS]"
  echo "  MODEL: HuggingFace model identifier (e.g., meta-llama/Llama-3.1-8B-Instruct)"
  echo "  TP_SIZE: Tensor parallelism size (e.g., 1, 2, 4, 8)"
  echo ""
  echo "Options:"
  echo "  --prompt <text>          Input prompt (default: 'Hello, how are you?')"
  echo "  --num-tokens <n>         Number of tokens to generate (default: 50)"
  echo "  --num-runs <n>           Number of measurement runs (default: 1)"
  echo "  --warmup-runs <n>        Number of warmup runs (default: 0)"
  echo "  --output <file>          Output JSON file path"
  echo ""
  echo "Example:"
  echo "  ./scripts/get_tpot.sh meta-llama/Llama-3.1-8B-Instruct 4 --num-runs 3"
  exit 1
fi

MODEL=$1
TP_SIZE=$2
shift 2

python3 ./scripts/measure_tpot.py --model "$MODEL" --tp-size "$TP_SIZE" "$@"

