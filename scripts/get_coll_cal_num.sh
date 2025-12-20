#!/bin/bash

# Example script to calculate NCCL collective call count
# Usage: ./scripts/get_coll_cal_num.sh [trace_directory]

TRACE_DIR="${1:-./trace_collection/llama3.1-8b-torchtitan-tp-perlmutter-16}"

python -m tools.main --trace "${TRACE_DIR}" --metric "coll_call_num-16"
