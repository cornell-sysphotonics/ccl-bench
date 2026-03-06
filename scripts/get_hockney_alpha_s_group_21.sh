#!/bin/bash

# Group 21 traces with number of chips
traces=(
  "group-21/llama-3.1-8b-torchxla_tp_v6e-4-tpu-group_21.trace.json:4"
  "group-21/llama-3.1-8b-torchxla_tp_v6e-8-tpu-group_21.trace.json:8"
  "group-21/llama-3.1-8b-torchxla_fsdp_v6e-4-tpu-group_21.trace.json:4"
  "group-21/llama-3.1-8b-torchxla_fsdp_v6e-8-tpu-group_21.trace.json:8"
  "group-21/llama-3.1-8b-torchxla_hybrid-fsdp-tp_v6e-4-tpu-group_21.trace.json:4"
  "group-21/llama-3.1-8b-torchxla_hybrid-fsdp-tp_v6e-8-tpu-group_21.trace.json:8"
)

for trace_spec in "${traces[@]}"; do
  trace_path="${trace_spec%%:*}"
  n_chips="${trace_spec##*:}"
  echo "===== $trace_path (n_chips=$n_chips) ====="
  python ./tools/main.py --trace "$trace_path" --metric "hockney_alpha_s" --n_chips "$n_chips"
  echo ""
done
