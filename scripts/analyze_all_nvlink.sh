#!/bin/bash
#
# Analyze NVLink throughput for all experiment directories in $PSCRATCH/nvlink_experiments
#
# Outputs to results/<experiment>/ directory with:
# - nvlink_metrics.txt: Max, average, and total communication metrics
#
# Usage:
#   ./analyze_all_nvlink.sh                    # Analyze all experiments
#   ./analyze_all_nvlink.sh --link 0           # Filter to link 0
#   ./analyze_all_nvlink.sh --direction tx     # Filter to TX only
#

# Script directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to the analysis script
ANALYZE_SCRIPT="$REPO_ROOT/tools/nvlink_usage/analyze_nvlink_throughput.py"

# Base directory for experiments
EXPERIMENTS_DIR="${PSCRATCH}/nvlink_experiments"

# Output directory
OUTPUT_DIR="$REPO_ROOT/results"
mkdir -p "$OUTPUT_DIR"

# Check if PSCRATCH is set
if [ -z "$PSCRATCH" ]; then
    echo "ERROR: PSCRATCH environment variable is not set"
    exit 1
fi

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "ERROR: Experiments directory does not exist: $EXPERIMENTS_DIR"
    exit 1
fi

# Check if analysis script exists
if [ ! -f "$ANALYZE_SCRIPT" ]; then
    echo "ERROR: Analysis script not found: $ANALYZE_SCRIPT"
    exit 1
fi

# Parse arguments to pass through to the analysis script
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    EXTRA_ARGS="$EXTRA_ARGS $1"
    shift
done

echo "============================================================"
echo "NVLink Throughput Analysis - All Experiments"
echo "============================================================"
echo ""
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Analysis script: $ANALYZE_SCRIPT"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra arguments: $EXTRA_ARGS"
fi
echo ""

# Find all directories that contain nvlink_trace.bin
EXPERIMENT_DIRS=()
while IFS= read -r -d '' dir; do
    EXPERIMENT_DIRS+=("$(dirname "$dir")")
done < <(find "$EXPERIMENTS_DIR" -name "nvlink_trace.bin" -print0 2>/dev/null)

if [ ${#EXPERIMENT_DIRS[@]} -eq 0 ]; then
    echo "No directories with nvlink_trace.bin found in $EXPERIMENTS_DIR"
    exit 0
fi

echo "Found ${#EXPERIMENT_DIRS[@]} experiment(s) with NVLink traces:"
for dir in "${EXPERIMENT_DIRS[@]}"; do
    echo "  - $(basename "$dir")"
done
echo ""

# Process each experiment directory
SUCCESS_COUNT=0
FAIL_COUNT=0

for exp_dir in "${EXPERIMENT_DIRS[@]}"; do
    exp_name=$(basename "$exp_dir")
    
    echo "Processing: $exp_name"
    
    # Create experiment output directory
    exp_output_dir="$OUTPUT_DIR/$exp_name"
    mkdir -p "$exp_output_dir"
    
    # Run analysis and save output
    output_file="$exp_output_dir/nvlink_metrics.txt"
    
    if python3 "$ANALYZE_SCRIPT" "$exp_dir" --metrics all $EXTRA_ARGS > "$output_file" 2>&1; then
        echo "  Output: $output_file"
        # Show a brief summary
        grep -E "(Max Throughput|Avg Throughput|Total GB):" "$output_file" | head -3 | sed 's/^/  /'
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ERROR: Analysis failed for $exp_name"
        cat "$output_file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
done

echo "============================================================"
echo "Summary"
echo "============================================================"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "  Total: ${#EXPERIMENT_DIRS[@]}"
echo ""
echo "Results saved to: $OUTPUT_DIR/<experiment>/nvlink_metrics.txt"
echo ""
echo "Done!"
