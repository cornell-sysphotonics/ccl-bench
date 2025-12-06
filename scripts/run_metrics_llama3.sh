#!/usr/bin/env bash
# Run all metrics for LLaMA-3.1-8B workload
# Usage: ./run_metrics_llama3.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$PROJECT_ROOT/tools"
TRACE_DIR="$PROJECT_ROOT/trace_collection/llama3.1-8b-torchtitan-pp-perlmutter-16"
OUTPUT_CSV="$PROJECT_ROOT/metrics_llama3.csv"

# Available metrics
METRICS=(
	"coll_call_num"
	"throughput_tokens"
	"iter_time"
	"comm_comp_overlap"
	"pipeline_bubble"
	"straggler_lag"
)

echo "CCL-Bench Metrics for LLaMA-3.1-8B"
echo "=================================="
echo "Trace directory: $TRACE_DIR"
echo ""

# Check if trace directory exists
if [ ! -d "$TRACE_DIR" ]; then
	echo "Warning: Trace directory not found: $TRACE_DIR"
	echo "Creating placeholder directory..."
	mkdir -p "$TRACE_DIR"
fi

# Write CSV header
echo "metric,value" > "$OUTPUT_CSV"

# Run each metric
cd "$TOOLS_DIR"
for metric in "${METRICS[@]}"; do
	echo "Calculating: $metric"
	value=$(python main.py --trace "$TRACE_DIR" --metric "$metric" 2> /dev/null || echo "N/A")
	echo "$metric,$value" >> "$OUTPUT_CSV"
	echo "  Result: $value"
done

echo ""
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "CSV Contents:"
cat "$OUTPUT_CSV"
