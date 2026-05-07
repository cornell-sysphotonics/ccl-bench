#!/bin/bash
# Run script for llama-3.1-8b-torchxla_tp_v6e-8-tpu-group_21
# Framework: TorchXLA  |  Model: llama-3.1-8b  |  TP=8  FSDP=1
# Usage: $0 <ZONE> <TPU_NAME>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME>" >&2
  exit 1
fi

ZONE=$1
TPU_NAME=$2
MODEL="llama-3.1-8b"
TP=8
FSDP=1
BATCH_SIZE=1
SEQ_LEN=512

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
