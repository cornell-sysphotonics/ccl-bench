#!/usr/bin/env bash
set -euo pipefail

# Gloo SGLang Kineto/NSYS runs for batch=128 variants.
#
# Follows the same approach as vLLM's Gloo script:
#   - COMM_BACKEND=gloo sets CCL_BENCH_DIST_BACKEND=gloo
#   - disable-custom-all-reduce is set to force torch.distributed path

export TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-gloo-batch128}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-gloo-batch128-bundles}"
export VARIANTS="qwen_b128 llama_b128 deepseek_b128"
export COMM_BACKEND="gloo"

bash scripts/run_sglang_inference_perlmutter.sh
bash scripts/make_sglang_inference_bundles_perlmutter.sh
