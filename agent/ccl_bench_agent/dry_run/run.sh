#!/usr/bin/env bash
# Run the CCL-Search dry-run example.
#
# No GPU, cluster, or real workload required.  The mock run script generates
# synthetic kineto traces; step time varies with TP/DP/PP choices so the agent
# gets meaningful feedback and converges toward the optimal config.
#
# Expected optimum: tp=4, dp=1, pp=1, micro_batch=1, act_ckpt=false → ~0.31 s
#
# Usage (from anywhere):
#   bash agent/ccl_bench_agent/dry_run/run.sh
set -euo pipefail

# agent.py imports siblings (execute, compute_metric, …) by relative name,
# so it must be invoked from the ccl_bench_agent/ directory.
AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AGENT_DIR"

python agent.py \
  --card     dry_run/workload_card.yaml \
  --tuning   dry_run/tuning_config.yaml \
  --seed     generate_config.py \
  --max-iterations 10 \
  --patience  4
