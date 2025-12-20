#!/bin/bash

python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_1_tp_4_ep_4-default-noCUgraph-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_1_tp_4_ep_4-naive-noCUgraph-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_2_tp_2_ep_4-default-noCUgraph-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_2_tp_2_ep_4-naive-noCUgraph-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-default-noCUgraph-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-default-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-naive-noCUgraph-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-naive-perlmutter-group_9" --metric "compute_bound_fraction"
python ./tools/main.py --trace "./trace_collection/deepseek-v2-lite-vllm-dp_4_tp_1_ep_4-pplx-perlmutter-group_9" --metric "compute_bound_fraction"
