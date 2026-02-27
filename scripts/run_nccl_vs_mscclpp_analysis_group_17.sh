#!/bin/bash
# Script to compare NCCL and MSCCL++ trace performance
# Usage: ./run_nccl_vs_mscclpp_analysis.sh <path-to-nccl-trace> <path-to-mscclpp-trace>
# Example:
#   ./run_nccl_vs_mscclpp_analysis.sh group-17/llama-3.1-8b-kineto_trace_nccl_DDP-group17.json group-17/llama-3.1-8b-kineto_trace_mscclpp_DDP-group17.json

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path-to-nccl-trace> <path-to-mscclpp-trace>"
    exit 1
fi

NCCL_TRACE=$1
MSCClPP_TRACE=$2
OUTPUT_DIR="./tools/throughput_tokens_sec,Comm_overhead-group17/group17-pipeline/results"

python ./tools/throughput_tokens_sec,Comm_overhead-group17/group17-pipeline/ana.py --nccl-trace "$NCCL_TRACE" --mscclpp-trace "$MSCClPP_TRACE" --output-dir "$OUTPUT_DIR"

echo "\nResults saved to:"
echo "  $OUTPUT_DIR/analysis_results.json      # Main metrics (throughput, overhead, bandwidth, etc.)"
echo "  $OUTPUT_DIR/performance_comparison.png # Comparison chart"
echo "  $OUTPUT_DIR/performance_report.html    # HTML summary report"


# # run ana tool with this command:
# python ./tools/throughput_tokens_sec,Comm_overhead-group17/group17-pipeline/ana.py --nccl-trace <path-to-nccl-trace> --mscclpp-trace <path-to-mscclpp-trace> --output-dir ./tools/throughput_tokens_sec,Comm_overhead-group17/group17-pipeline/results

# example:
# python ./tools/throughput_tokens_sec,Comm_overhead-group17/group17-pipeline/ana.py --nccl-trace group-17/llama-3.1-8b-kineto_trace_nccl_DDP-group17.json --mscclpp-trace group-17/llama-3.1-8b-kineto_trace_mscclpp_DDP-group17.json --output-dir ./tools/throughput_tokens_sec,Comm_overhead-group17/group17-pipeline/results