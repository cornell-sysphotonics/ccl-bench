#!/usr/bin/env bash
set -euo pipefail

# Pure MSCCL++ SGLang Kineto/NSYS runs for batch=128 variants.
#
# Follows the same approach as vLLM's pure MSCCL++ script:
#   - COMM_BACKEND=pure_mscclpp uses LD_PRELOAD and SGLANG_NCCL_SO_PATH
#   - disable-custom-all-reduce is set to force torch.distributed path

export TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-pure-mscclpp-batch128}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-pure-mscclpp-batch128-bundles}"
export VARIANTS="qwen_b128 llama_b128 deepseek_b128"
export COMM_BACKEND="pure_mscclpp"

bash scripts/run_sglang_inference_perlmutter.sh
bash scripts/make_sglang_inference_bundles_perlmutter.sh
