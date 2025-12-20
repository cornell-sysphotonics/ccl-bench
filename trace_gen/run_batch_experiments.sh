#!/bin/bash
#
# Run NVLink profiling experiments with varying batch sizes
#
# Usage:
#   ./run_batch_experiments.sh
#   BATCH_SIZES="1 2" ./run_batch_experiments.sh
#   BASE_DIR=/custom/path ./run_batch_experiments.sh
#

set -e

BASE_DIR="${BASE_DIR:-$SCRATCH/nvlink_batch_experiments}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"

echo "========================================"
echo "=== NVLink Batch Size Experiments ==="
echo "========================================"
echo "Base directory: $BASE_DIR"
echo "Batch sizes: $BATCH_SIZES"
echo "Started: $(date)"
echo ""

mkdir -p "$BASE_DIR"

# Save experiment config
cat > "$BASE_DIR/experiment_config.json" << EOF
{
  "batch_sizes": "$BATCH_SIZES",
  "started": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "base_dir": "$BASE_DIR"
}
EOF

for bs in $BATCH_SIZES; do
    echo ""
    echo "========================================"
    echo "[$(date)] Starting experiment: batch_size=$bs"
    echo "========================================"
    
    export MAX_NUM_SEQS=$bs
    export MAX_BATCHED_TOKENS=$((bs * 4096))
    export NUM_PROMPTS=$((bs * 5))
    export REQUEST_RATE=inf
    export OUTPUT_DIR="$BASE_DIR/batch_$bs"
    
    echo "Configuration:"
    echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS"
    echo "  MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS"
    echo "  NUM_PROMPTS=$NUM_PROMPTS"
    echo "  REQUEST_RATE=$REQUEST_RATE"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
    echo ""
    
    # Run the profiling script
    ./nsys_profile_server.sh
    
    # Save per-experiment config
    cat > "$OUTPUT_DIR/batch_config.json" << EOF
{
  "batch_size": $bs,
  "max_num_seqs": $MAX_NUM_SEQS,
  "max_batched_tokens": $MAX_BATCHED_TOKENS,
  "num_prompts": $NUM_PROMPTS,
  "request_rate": "$REQUEST_RATE",
  "completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    echo ""
    echo "[$(date)] Completed: batch_size=$bs"
    echo "  Output: $OUTPUT_DIR"
    echo ""
done

# Update experiment config with completion time
cat > "$BASE_DIR/experiment_config.json" << EOF
{
  "batch_sizes": "$BATCH_SIZES",
  "started": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "base_dir": "$BASE_DIR"
}
EOF

echo ""
echo "========================================"
echo "=== All Experiments Complete ==="
echo "========================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to: $BASE_DIR"
echo ""
ls -la "$BASE_DIR"/
echo ""
echo "Directory sizes:"
du -sh "$BASE_DIR"/batch_*/ 2>/dev/null || true
echo ""
echo "To analyze results, run:"
for bs in $BATCH_SIZES; do
    echo "  python3 ../tools/nvlink/correlate_nsys_nvlink.py $BASE_DIR/batch_$bs"
done

