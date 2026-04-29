#!/bin/bash
# run_agent.sh — Launch the CCL-Bench ADRS loop for Perlmutter/torchtitan/Llama-3.1-8B
#
# Usage:
#   bash run_agent.sh [--max-iterations N] [--patience N]
#
# Prerequisites:
#   1. Set your Anthropic API key:
#        echo "sk-ant-..." > /pscratch/sd/e/ericding/ccl-bench/agent/API_KEY
#      Or export ANTHROPIC_API_KEY=sk-ant-... before running.
#
#   2. The Opus venv at /pscratch/sd/e/ericding/Opus/.venv must be accessible.
#      anthropic and pyyaml are installed automatically on first run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/../ccl_bench_agent"
# VENV="$PSCRATCH/Opus/.venv"
API_KEY_FILE="$SCRIPT_DIR/../API_KEY"

MAX_ITERATIONS=${MAX_ITERATIONS:-15}
PATIENCE=${PATIENCE:-5}

# Allow CLI overrides: --max-iterations N and --patience N
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --patience)       PATIENCE="$2";       shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Activate venv ─────────────────────────────────────────────────────────────
# if [ ! -f "$VENV/bin/activate" ]; then
#     echo "ERROR: venv not found at $VENV" >&2
#     echo "  Update the VENV variable in this script to point to your Python env." >&2
#     exit 1
# fi

# source "$VENV/bin/activate"

# ── Install missing agent deps (no-op after first run) ────────────────────────
pip install --quiet --no-warn-script-location anthropic pyyaml

# ── API key: file takes priority; fall back to env var ────────────────────────
if [ ! -f "$API_KEY_FILE" ]; then
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        echo "$ANTHROPIC_API_KEY" > "$API_KEY_FILE"
        echo "[run_agent] Wrote API key from \$ANTHROPIC_API_KEY to $API_KEY_FILE"
    else
        echo "ERROR: No API key found." >&2
        echo "  Either:" >&2
        echo "    echo 'sk-ant-...' > $API_KEY_FILE" >&2
        echo "  or:" >&2
        echo "    export ANTHROPIC_API_KEY=sk-ant-..." >&2
        exit 1
    fi
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  CCL-Bench ADRS — Perlmutter · torchtitan · Llama-3.1-8B"
echo "  max_iterations=$MAX_ITERATIONS  patience=$PATIENCE"
echo "================================================================"

cd "$AGENT_DIR"
exec python agent.py \
    --card    "$SCRIPT_DIR/workload_card.yaml" \
    --tuning  "$SCRIPT_DIR/tuning_config.yaml" \
    --seed    "$SCRIPT_DIR/generate_config.py" \
    --max-iterations "$MAX_ITERATIONS" \
    --patience       "$PATIENCE"
