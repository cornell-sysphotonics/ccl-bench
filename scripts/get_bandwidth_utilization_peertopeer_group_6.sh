#!/bin/bash

traces=(
  "./trace_collection/qwen-32b-sglang-pp_1-perlmutter-group_6"
  "./trace_collection/qwen-32b-sglang-pp_2-perlmutter-group_6"
  "./trace_collection/deepseek-v2-lite-sglang-ep_1-perlmutter-group_6"
  "./trace_collection/deepseek-v2-lite-sglang-ep_2-perlmutter-group_6"
  "./trace_collection/deepseek-v2-lite-sglang-ep_4-perlmutter-group_6"
  "./trace_collection/llama-3.1-8b-sglang-tp_1-pp_4-perlmutter-group_6"
  "./trace_collection/llama-3.1-8b-sglang-tp_2-pp_2-perlmutter-group_6"
  "./trace_collection/llama-3.1-8b-sglang-tp_4-pp_1-perlmutter-group_6"
)

for trace in "${traces[@]}"; do
  python ./tools/main.py --trace "$trace" --metric "bandwidth_utilization_peertopeer_group_6"
done
