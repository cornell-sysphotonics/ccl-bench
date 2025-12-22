#!/bin/bash
set -euo pipefail

ROOT_DIR="/pscratch/sd/b/bck/inference_sweep_output"
METRIC_SCRIPT="$HOME/ccl-bench-straggler-scale-up/tools/main.py"

# Iterate over subdirectories of ROOT_DIR
for trace_dir in "${ROOT_DIR}"/*/; do
    [ -d "$trace_dir" ] || continue

    echo "=== Processing directory: $trace_dir ==="

    shopt -s nullglob
    gz_files=("$trace_dir"*.gz)

    if ((${#gz_files[@]} > 0)); then
        for gz in "${gz_files[@]}"; do
            json_path="${gz%.gz}"

            if [ -f "$json_path" ]; then
                echo "  Skipping $gz (JSON already exists)"
            else
                echo "  Decompressing $gz -> $json_path"
                gunzip -c "$gz" > "$json_path"
            fi
        done
    else
        echo "  No .gz files found in $trace_dir"
    fi
    shopt -u nullglob

    echo "  Running straggler_metrics on $trace_dir"
    python "$METRIC_SCRIPT" --trace "$trace_dir" --metric straggler_metrics || {
        echo "  WARNING: metric script failed for $trace_dir"
    }

    echo
done