#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <trace_directory>"
  exit 1
fi

trace_dir="$1"

for trace_path in "$trace_dir"/*.trace.json; do
  echo "===== $trace_path ====="
  python ./tools/main.py --trace "$trace_path" --metric "avg_comm_bandwidth_GBps"
  echo ""
done
