#!/bin/bash
# Baseline simulation: deepseek-v3-16b ep4-dp2-tp4 on A100 + Slingshot.
# 8 ranks, no pipeline parallelism → clean collective-only trace.
# Two-tier hardware: 4 GPUs/node NVLink scale-up + Slingshot scale-out.

set -e
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE=/data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter
OUTDIR="$REPO/simulation/examples/01_baseline_output"

GPUS_PER_NODE=4
# INTRA_BW=300
# INTRA_LAT=50
# INTER_BW=25
# INTER_LAT=500

INTRA_BW=600
INTRA_LAT=50
INTER_BW=50
INTER_LAT=500

python "$REPO/simulation/pipeline.py" \
    --mode comm-only \
    --trace-dir "$TRACE" \
    --output-dir "$OUTDIR" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --intra-topology FullyConnected \
    --intra-bandwidth "$INTRA_BW" \
    --intra-latency "$INTRA_LAT" \
    --topology Switch \
    --bandwidth "$INTER_BW" \
    --latency "$INTER_LAT" \
    --collective-algo ring \
    --compute-model kernels

echo
echo "Outputs in: $OUTDIR"
