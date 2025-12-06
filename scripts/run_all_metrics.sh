#!/usr/bin/env bash
# Run all metrics for all workloads and combine into a single report
# Usage: ./run_all_metrics.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$PROJECT_ROOT/tools"
OUTPUT_CSV="$PROJECT_ROOT/metrics_all_workloads.csv"

# Workloads to process
WORKLOADS=(
	"llama3.1-8b-torchtitan-pp-perlmutter-16"
	"llama3.1-8b-torchtitan-tp-perlmutter-16"
	"deepseek-v2-lite-torchtitan-dp+pp-perlmutter-16"
	"deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16"
	"qwen3-32b-torchtitan-3d-perlmutter-16"
	"qwen3-32b-torchtitan-dp+pp-perlmutter-16"
	"qwen3-32b-torchtitan-dp+tp-perlmutter-16"
)

# Available metrics
METRICS=(
	"coll_call_num"
	"throughput_tokens"
	"iter_time"
	"comm_comp_overlap"
	"pipeline_bubble"
	"straggler_lag"
)

echo "CCL-Bench Metrics - All Workloads"
echo "================================="
echo ""

# Write CSV header
echo "workload,metric,value" > "$OUTPUT_CSV"

cd "$TOOLS_DIR"

for workload in "${WORKLOADS[@]}"; do
	TRACE_DIR="$PROJECT_ROOT/trace_collection/$workload"
	echo "Processing: $workload"
	echo "  Trace directory: $TRACE_DIR"

	# Check if trace directory exists
	if [ ! -d "$TRACE_DIR" ]; then
		echo "  Warning: Trace directory not found, skipping..."
		continue
	fi

	for metric in "${METRICS[@]}"; do
		echo "  Calculating: $metric"
		value=$(python main.py --trace "$TRACE_DIR" --metric "$metric" 2> /dev/null || echo "N/A")
		echo "$workload,$metric,$value" >> "$OUTPUT_CSV"
	done
	echo ""
done

echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
cat "$OUTPUT_CSV"
