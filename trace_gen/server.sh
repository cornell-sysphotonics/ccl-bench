#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,INIT

python3 -m vllm.entrypoints.openai.api_server \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --model Qwen/Qwen3-32B \
  --swap-space 16 \
  --disable-log-requests \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --disable-sliding-window
> logs/server.log

