#!/usr/bin/env bash
# Analyze workload traces and generate reports
# Usage: ./analyze_traces.sh <trace_base_dir> [output_dir] [workload_name]
#
# Examples:
#   ./analyze_traces.sh /pscratch/sd/i/imh39/ccl-bench-traces/llama3_8b_pp/torch_traces
#   ./analyze_traces.sh /path/to/traces ./my_output "My Workload Name"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TOOLS_DIR="${PROJECT_ROOT}/tools"

# Parse arguments
TRACE_BASE="${1:-}"
OUTPUT_DIR="${2:-}"
WORKLOAD_NAME="${3:-}"

if [[ -z ${TRACE_BASE} ]]; then
	echo "Usage: $0 <trace_base_dir> [output_dir] [workload_name]"
	echo ""
	echo "Arguments:"
	echo "  trace_base_dir  Directory containing iteration_* subdirectories"
	echo "  output_dir      Output directory (default: auto-generated in analysis_output/)"
	echo "  workload_name   Name for the workload (default: auto-detected)"
	echo ""
	echo "Examples:"
	echo "  $0 /pscratch/sd/i/imh39/ccl-bench-traces/llama3_8b_pp/torch_traces"
	echo "  $0 ./traces ./output 'My Model PP'"
	exit 1
fi

# Check trace directory exists
if [[ ! -d ${TRACE_BASE} ]]; then
	echo "Error: Trace directory not found: ${TRACE_BASE}"
	exit 1
fi

# Auto-generate output directory if not provided
if [[ -z ${OUTPUT_DIR} ]]; then
	# Extract workload name from path
	PARENT_DIR=$(basename "$(dirname "${TRACE_BASE}")")
	OUTPUT_DIR="${PROJECT_ROOT}/analysis_output/${PARENT_DIR}"
fi

# Build command
CMD="python3 ${TOOLS_DIR}/analyze_workload.py --trace-base \"${TRACE_BASE}\" --output \"${OUTPUT_DIR}\""

if [[ -n ${WORKLOAD_NAME} ]]; then
	CMD="${CMD} --name \"${WORKLOAD_NAME}\""
fi

echo "========================================"
echo "CCL-Bench Workload Analysis"
echo "========================================"
echo "Trace Base: ${TRACE_BASE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo ""

# Run analysis
eval "${CMD}"

echo ""
echo "========================================"
echo "Analysis Complete!"
echo "========================================"
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}"
echo ""
echo "To view the HTML report, open:"
echo "  ${OUTPUT_DIR}/report.html"
