#!/bin/bash
# Run script for llama-3.1-8b-torchxla_fsdp_v6e-4-tpu-group_21
# Framework: TorchXLA  |  Model: llama-3.1-8b  |  TP=1  FSDP=4
# Usage: $0 <ZONE> <TPU_NAME>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ZONE> <TPU_NAME>" >&2
  exit 1
fi

ZONE=$1
TPU_NAME=$2
MODEL="llama-3.1-8b"
TP=1
FSDP=4
BATCH_SIZE=4
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
