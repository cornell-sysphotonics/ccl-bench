#!/usr/bin/env bash
# CCL-Search dry-run mock run script.
#
# Called by execute.py as the workload's run_script.  Receives config choices
# as uppercase environment variables (e.g. TP=4, DP=1) and TRACE_DIR pointing
# to where the trace should land.
#
# No GPU, cluster, or network connection required.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/mock_trace_gen.py"
