#!/bin/bash

python ./tools/main.py --trace "/data/ccl-bench_trace_collection/Llama-3.1-8B-torchxla-vllm-tp1-tpu-group-4" --metric "avg_step_time"

python ./tools/main.py --trace "/data/ccl-bench_trace_collection/Llama-3.1-8B-torchxla-vllm-tp2-tpu-group-4" --metric "avg_step_time"

python ./tools/main.py --trace "/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep8-dp8-perlmutter" --metric "avg_step_time"