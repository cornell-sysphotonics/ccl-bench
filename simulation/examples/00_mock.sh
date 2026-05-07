#!/bin/bash
# CCL-Bench simulation dry run — no Docker, no AstraSim, no real traces.
#
# Generates synthetic rankN_trace.json files for Llama-3.1-8B training
# (tp=4, dp=2, 8 GPUs) then runs the analytical pipeline to estimate
# step time and comm fraction under different network configurations.
#
# Usage: bash simulation/examples/00_mock.sh
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TRACE_DIR="/tmp/ccl_bench_mock_trace"

echo "=== Step 1: Generate synthetic traces ==="
python "$REPO/simulation/mock_trace_gen.py" \
    --output-dir "$TRACE_DIR" \
    --tp 4 --dp 2 \
    --layers 32 --hidden-size 4096 \
    --seq-len 1024 --batch-size 4 \
    --dtype BFloat16

echo
echo "=== Step 2: Baseline (A100 NVLink + Slingshot 25 GB/s) ==="
python "$REPO/simulation/mock_pipeline.py" \
    --trace-dir "$TRACE_DIR" \
    --gpus-per-node 4 \
    --intra-bandwidth 400 --intra-latency 50 \
    --bandwidth 25 --latency 5000

echo
echo "=== Step 3: What-if — 2× scale-out bandwidth (50 GB/s) ==="
python "$REPO/simulation/mock_pipeline.py" \
    --trace-dir "$TRACE_DIR" \
    --gpus-per-node 4 \
    --intra-bandwidth 400 --intra-latency 50 \
    --bandwidth 50 --latency 5000

echo
echo "=== Step 4: What-if — H100-class NVLink (900 GB/s intra) ==="
python "$REPO/simulation/mock_pipeline.py" \
    --trace-dir "$TRACE_DIR" \
    --gpus-per-node 4 \
    --intra-bandwidth 900 --intra-latency 30 \
    --bandwidth 25 --latency 5000

echo
echo "Traces written to: $TRACE_DIR"
echo "Run with real AstraSim (requires Docker):"
echo "  python $REPO/simulation/pipeline.py --mode comm-only --trace-dir $TRACE_DIR"
