#!/bin/bash

python ./tools/main.py --trace "<path-to-trace>/deepseek_r1_distill_qwen_7b-megatron_lm-dp_2-tp_4-pp_2-perlmutter-group_1" --metric "traffic_window"

python ./tools/main.py --trace "<path-to-trace>/deepseek_r1_distill_qwen_7b-megatron_lm-dp_4-tp_4-perlmutter-group_1" --metric "traffic_window"

python ./tools/main.py --trace "<path-to-trace>/llama_31_8b-megatron_lm-dp_2-tp_4-perlmutter-group_1" --metric "traffic_window"

python ./tools/main.py --trace "<path-to-trace>/llama_31_8b-megatron_lm-dp_4-tp_2-perlmutter-group_1" --metric "traffic_window"

python ./tools/main.py --trace "<path-to-trace>/mistral_7b_instruct_v02-megatron_lm-dp_2-tp_2-pp_2-perlmutter-group_1" --metric "traffic_window"

python ./tools/main.py --trace "<path-to-trace>/mistral_7b_instruct_v02-megatron_lm-dp_2-tp_4-perlmutter-group_1" --metric "traffic_window"