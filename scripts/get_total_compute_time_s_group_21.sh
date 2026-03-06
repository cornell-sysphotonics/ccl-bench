#!/bin/bash

# Group 21 traces
traces=(
  "group-21/llama-3.1-8b-torchxla_tp_v6e-4-tpu-group_21.trace.json"
  "group-21/llama-3.1-8b-torchxla_tp_v6e-8-tpu-group_21.trace.json"
  "group-21/llama-3.1-8b-torchxla_fsdp_v6e-4-tpu-group_21.trace.json"
  "group-21/llama-3.1-8b-torchxla_fsdp_v6e-8-tpu-group_21.trace.json"
  "group-21/llama-3.1-8b-torchxla_hybrid-fsdp-tp_v6e-4-tpu-group_21.trace.json"
  "group-21/llama-3.1-8b-torchxla_hybrid-fsdp-tp_v6e-8-tpu-group_21.trace.json"
)

for trace in "${traces[@]}"; do
  echo "===== $trace ====="
  python ./tools/main.py --trace "$trace" --metric "total_compute_time_s"
  echo ""
done
