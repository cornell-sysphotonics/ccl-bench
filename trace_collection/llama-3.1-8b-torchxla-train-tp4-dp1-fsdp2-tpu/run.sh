#!/bin/bash
# Run script for llama-3.1-8b-torchxla-train-tp4-dp1-fsdp2-tpu
# Framework: TorchXLA  |  Model: llama-3.1-8b  |  TP=4  FSDP=2
# Usage: $0 <ZONE> <TPU_NAME>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME>" >&2
  exit 1
fi

ZONE=$1
TPU_NAME=$2
MODEL="llama-3.1-8b"
TP=4
FSDP=2
BATCH_SIZE=32
SEQ_LEN=1024

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone "$ZONE" \
  --command "
    cd ~/
    python3 train_llm_xla.py \
      --model $MODEL \
      --tensor-parallel $TP \
      --fsdp $FSDP \
      --batch-size $BATCH_SIZE \
      --seq-len $SEQ_LEN \
      --profile
  "
