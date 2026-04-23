#!/usr/bin/env bash
set -euo pipefail

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-gloo-batch128}"
LOG_DIR="${LOG_DIR:-$TRACE_ROOT}"
mkdir -p "$LOG_DIR"

salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account m4999 \
  srun --nodes 1 --ntasks 1 --gpus 4 --cpus-per-task 64 --gpu-bind=none bash -lc '
    set -euo pipefail
    source ~/.bashrc >/dev/null 2>&1 || true
    module load conda >/dev/null 2>&1
    conda activate /pscratch/sd/k/kg597/session1/cvllm
    cd ~/ccl-bench
    export TRACE_ROOT="'"$TRACE_ROOT"'"
    export BUNDLE_ROOT="${PSCRATCH}/ccl-bench-traces/llama-deepseek-gloo-batch128-bundles"
    bash scripts/vllm_gloo_patch/install.sh
    bash scripts/run_llama_deepseek_gloo_batch128_perlmutter.sh
    bash scripts/make_llama_deepseek_gloo_batch128_bundles_perlmutter.sh
  '
