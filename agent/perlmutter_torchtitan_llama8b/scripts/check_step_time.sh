#!/bin/bash
# Manually compute avg_step_time for the Perlmutter TorchTitan Llama 8B traces.
#
# Usage:
#   ./check_step_time.sh
#   ./check_step_time.sh /path/to/trace_dir
#
# The TorchTitan run script usually flattens rank JSON traces into TRACE_DIR.
# This helper also handles the nested profile_traces/iteration_N layout. It
# checks one rank trace only, which is enough for a quick manual sanity check.

set -euo pipefail

DEFAULT_TRACE_DIR="/pscratch/sd/e/ericding/ccl-bench/perlmutter_llama8b"
TRACE_DIR="${1:-$DEFAULT_TRACE_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TOOLS_MAIN="$REPO_ROOT/tools/main.py"

if [ ! -d "$TRACE_DIR" ]; then
    echo "Trace directory does not exist: $TRACE_DIR" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ccl_step_time.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

copy_or_link_yaml() {
    local yaml_file
    yaml_file="$(find "$TRACE_DIR" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | head -1)"
    if [ -n "$yaml_file" ]; then
        ln -s "$yaml_file" "$TMP_DIR/$(basename "$yaml_file")"
    else
        cat > "$TMP_DIR/trace_meta.yaml" <<'YAML'
metric_source:
  traces:
    - json
YAML
    fi
}

link_one_json_file() {
    local source_dir="$1"
    local json_file

    json_file="$(find "$source_dir" -maxdepth 1 -type f -name '*.json' | sort | head -1)"
    if [ -n "$json_file" ]; then
        ln -s "$json_file" "$TMP_DIR/$(basename "$json_file")"
        echo "$json_file"
    fi
}

json_file="$(link_one_json_file "$TRACE_DIR")"

if [ -z "$json_file" ]; then
    ITER_DIR="$(find "$TRACE_DIR/profile_traces" -maxdepth 1 -type d -name 'iteration_*' 2>/dev/null | sort | tail -1)"
    if [ -n "$ITER_DIR" ]; then
        json_file="$(link_one_json_file "$ITER_DIR")"
    fi
fi

if [ -z "$json_file" ]; then
    echo "No JSON trace files found in $TRACE_DIR or $TRACE_DIR/profile_traces/iteration_*" >&2
    exit 1
fi

copy_or_link_yaml

echo "trace_dir: $TRACE_DIR"
echo "rank_trace: $json_file"
echo -n "avg_step_time_s: "
python3 "$TOOLS_MAIN" --trace "$TMP_DIR" --metric avg_step_time
