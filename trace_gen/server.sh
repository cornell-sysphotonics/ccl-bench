#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --model Qwen/Qwen3-32B \
    --swap-space 16 \
    --disable-log-requests \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --disable-sliding-window