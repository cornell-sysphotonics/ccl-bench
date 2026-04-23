#!/usr/bin/env bash
set -euo pipefail

# Pure MSCCL++ SGLang NSYS runs.
#
# Follows the same approach as vLLM's pure MSCCL++ script:
#   - COMM_BACKEND=pure_mscclpp uses LD_PRELOAD and SGLANG_NCCL_SO_PATH
#   - disable-custom-all-reduce is set to force torch.distributed path

export TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-pure-mscclpp}"
export SRC_ROOT="${SRC_ROOT:-$TRACE_ROOT}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/sglang-pure-mscclpp-bundles}"
export VARIANTS="${VARIANTS:-qwen_b8 qwen_b128 llama_b8 llama_b128 deepseek_b8 deepseek_b128}"
export COMM_BACKEND="pure_mscclpp"
export PROFILE_NSYS="${PROFILE_NSYS:-1}"
export NSYS_TRACE_TYPES="${NSYS_TRACE_TYPES:-cuda,nvtx}"
export NSYS_STOP_WAIT_SECONDS="${NSYS_STOP_WAIT_SECONDS:-600}"

bash scripts/run_sglang_inference_perlmutter.sh
bash scripts/make_sglang_inference_bundles_perlmutter.sh
