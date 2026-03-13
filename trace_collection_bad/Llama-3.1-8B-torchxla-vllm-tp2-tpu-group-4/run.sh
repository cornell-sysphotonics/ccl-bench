#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME>" >&2
  exit 1
fi

ZONE=$1
TPU_NAME=$2

# Ensure you have HF_TOKEN set in your environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/../.."

echo "Running meta-llama/Llama-3.1-8B TP=2 (All Batches)"
python3 $ROOT_DIR/scripts/run_trace-group4.py \
  --zone "$ZONE" \
  --tpu "$TPU_NAME" \
  --model "meta-llama/Llama-3.1-8B" \
  --tp 2
