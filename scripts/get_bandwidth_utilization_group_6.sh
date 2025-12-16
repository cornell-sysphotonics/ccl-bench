#!/bin/bash

traces=(
  "./trace_collection/qwen-32b-sglang-pp_1-perlmutter-group_6"
  "./trace_collection/qwen-32b-sglang-pp_2-perlmutter-group_6"
)

for trace in "${traces[@]}"; do
  python ./tools/main.py --trace "$trace" --metric "bandwidth_utilization_group_6"
done
