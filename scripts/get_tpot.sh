#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <trace_dir_name>"
  exit 1
fi

python ./tools/main.py --trace "./trace_collection/$1" --metric "tpot"
