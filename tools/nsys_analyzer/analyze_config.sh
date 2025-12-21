#!/bin/bash
# Configuration file for trace analysis
# Edit the TRACE_DIR variable to switch between different traces

TRACE_DIR="../../nsys"
# TRACE_DIR="../../nsys_1node_4gpu"
# TRACE_DIR="../../nsys_2node_8gpu"
# TRACE_DIR="/path/to/your/custom/nsys/directory"

WORKLOAD_NAME="llama-3.1-8b-deepspeed"

# 
# choices: nccl_calls,iteration_time,comm_breakdown,overlap,phase_windows,bandwidth
METRICS=""  # "" means run all metrics analysis
# METRICS="nccl_calls,iteration_time,comm_breakdown"

OUTPUT_JSON="results_${WORKLOAD_NAME}.json"
OUTPUT_CSV="all_results.csv"

echo "======================================================================="
echo "Trace Analysis Configuration"
echo "======================================================================="
echo "Trace directory: $TRACE_DIR"
echo "Workload name:   $WORKLOAD_NAME"
echo "Metrics:         ${METRICS:-all}"
echo "Output JSON:     $OUTPUT_JSON"
echo "Output CSV:      $OUTPUT_CSV"
echo ""

if [ ! -d "$TRACE_DIR" ]; then
    echo "Error: Trace directory not found: $TRACE_DIR"
    echo "Please edit analyze_config.sh and set the correct TRACE_DIR"
    exit 1
fi

if ! ls "$TRACE_DIR"/*.nsys-rep 1> /dev/null 2>&1; then
    echo "Error: No .nsys-rep files found in $TRACE_DIR"
    exit 1
fi

echo "Found nsys-rep files:"
ls -lh "$TRACE_DIR"/*.nsys-rep
echo ""

CMD="python3 analyze_trace.py \"$TRACE_DIR\" --name \"$WORKLOAD_NAME\""

if [ -n "$METRICS" ]; then
    CMD="$CMD --metrics \"$METRICS\""
fi

if [ -n "$OUTPUT_JSON" ]; then
    CMD="$CMD --output \"$OUTPUT_JSON\""
fi

if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --csv \"$OUTPUT_CSV\""
fi

echo "Running command:"
echo "$CMD"
echo ""
echo "======================================================================="
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✓ Analysis completed successfully!"
    echo "======================================================================="
    echo "Results saved to:"
    [ -f "$OUTPUT_JSON" ] && echo "  - $OUTPUT_JSON"
    [ -f "$OUTPUT_CSV" ] && echo "  - $OUTPUT_CSV"
else
    echo ""
    echo "======================================================================="
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "======================================================================="
fi

exit $EXIT_CODE

