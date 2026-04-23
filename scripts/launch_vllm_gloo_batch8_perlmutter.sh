#!/usr/bin/env bash
set -euo pipefail

# Collect the batch=8 Gloo rows for the same single-node TP4 vLLM inference
# workloads already covered by NCCL / NCCL+MSCCL++ / pure MSCCL++.

TRACE_ROOT_QWEN="${TRACE_ROOT_QWEN:-$PSCRATCH/ccl-bench-traces/qwen3-4b-gloo-batch8}"
BUNDLE_ROOT_QWEN="${BUNDLE_ROOT_QWEN:-$PSCRATCH/ccl-bench-traces/qwen3-4b-gloo-batch8-bundles}"
TRACE_ROOT_LD="${TRACE_ROOT_LD:-$PSCRATCH/ccl-bench-traces/llama-deepseek-gloo-batch8}"
BUNDLE_ROOT_LD="${BUNDLE_ROOT_LD:-$PSCRATCH/ccl-bench-traces/llama-deepseek-gloo-batch8-bundles}"
LOG_DIR="${LOG_DIR:-$PSCRATCH/ccl-bench-traces/vllm-gloo-batch8}"
mkdir -p "$LOG_DIR"

salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account m4999 \
  srun --nodes 1 --ntasks 1 --gpus 4 --cpus-per-task 64 --gpu-bind=none bash -lc '
    set -euo pipefail
    source ~/.bashrc >/dev/null 2>&1 || true
    module load conda >/dev/null 2>&1
    conda activate /pscratch/sd/k/kg597/session1/cvllm
    cd ~/ccl-bench
    bash scripts/vllm_gloo_patch/install.sh

    BATCH_SIZE=8 \
    TRACE_ROOT="'"$TRACE_ROOT_QWEN"'" \
      bash scripts/run_qwen3_4b_gloo_batch128_perlmutter.sh

    BATCH_SIZE=8 \
    SRC_ROOT="'"$TRACE_ROOT_QWEN"'" \
    BUNDLE_ROOT="'"$BUNDLE_ROOT_QWEN"'" \
      bash scripts/make_qwen3_4b_gloo_batch128_bundles_perlmutter.sh

    BATCH_SIZE=8 \
    TRACE_ROOT="'"$TRACE_ROOT_LD"'" \
      bash scripts/run_llama_deepseek_gloo_batch128_perlmutter.sh

    BATCH_SIZE=8 \
    SRC_ROOT="'"$TRACE_ROOT_LD"'" \
    BUNDLE_ROOT="'"$BUNDLE_ROOT_LD"'" \
      bash scripts/make_llama_deepseek_gloo_batch128_bundles_perlmutter.sh
  '
