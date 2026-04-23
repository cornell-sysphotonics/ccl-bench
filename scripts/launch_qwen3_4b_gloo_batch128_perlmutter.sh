#!/usr/bin/env bash
set -euo pipefail

# salloc + srun wrapper for the Qwen3-4B Gloo Perlmutter run.
#
# Usage:
#   bash scripts/launch_qwen3_4b_gloo_batch128_perlmutter.sh
#
# Gloo collectives over TCP are an order of magnitude slower than NCCL,
# so SALLOC_TIME defaults to 4 h. Reduce NUM_ITERS / PROFILE_WARMUP to
# shorten if needed.

TRACE_ROOT="${TRACE_ROOT:-/pscratch/sd/k/kg597/ccl-bench-traces/qwen3-4b-gloo-batch128}"
mkdir -p "$TRACE_ROOT"

salloc \
  --nodes 1 \
  --qos interactive \
  --time "${SALLOC_TIME:-04:00:00}" \
  --constraint gpu \
  --gpus 4 \
  --account m4999 \
  srun --nodes 1 --ntasks 1 --gpus 4 --cpus-per-task 64 --gpu-bind=none bash -lc '
    set -euo pipefail
    module load conda
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /pscratch/sd/k/kg597/session1/cvllm
    cd "$HOME/ccl-bench"
    mkdir -p /pscratch/sd/k/kg597/ccl-bench-traces/qwen3-4b-gloo-batch128
    bash scripts/run_qwen3_4b_gloo_batch128_perlmutter.sh
    bash scripts/make_qwen3_4b_gloo_batch128_bundles_perlmutter.sh
  '
