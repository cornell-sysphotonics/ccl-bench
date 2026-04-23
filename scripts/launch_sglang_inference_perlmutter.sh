#!/usr/bin/env bash
set -euo pipefail

# Perlmutter interactive launcher for the 6-row SGLang inference collection:
# Qwen3-4B, Llama-3.1-8B, DeepSeek-MoE-16B at batch 8 and 128.

TRACE_ROOT="${TRACE_ROOT:-/pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference-bundles}"
LOG_DIR="${LOG_DIR:-/pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ACCOUNT="${ACCOUNT:-m4999}"
QOS="${QOS:-interactive}"

mkdir -p "$LOG_DIR"

salloc \
  --nodes 1 \
  --qos "$QOS" \
  --time "$TIME_LIMIT" \
  --constraint gpu \
  --gpus 4 \
  --account "$ACCOUNT" \
  srun \
    --nodes 1 \
    --ntasks 1 \
    --gpus 4 \
    --cpus-per-task 64 \
    --gpu-bind=none \
    bash -lc "
      set -euo pipefail
      source ~/.bashrc >/dev/null 2>&1 || true
      module load conda >/dev/null 2>&1
      conda activate /pscratch/sd/k/kg597/session1/csglang
      cd ~/ccl-bench
      export TRACE_ROOT='$TRACE_ROOT'
      export BUNDLE_ROOT='$BUNDLE_ROOT'
      bash scripts/run_sglang_inference_perlmutter.sh
      bash scripts/make_sglang_inference_bundles_perlmutter.sh
    "
