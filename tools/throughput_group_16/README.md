# throughput_group_16

Computes tokens-per-second throughput using iteration timing from torch profile traces and tokens-per-step from a workload card.

- Inputs: iteration directory with torch Chrome trace JSON files; optional workload card (batch_size, seq_len) to set tokens_per_step.
- Outputs: throughput_tokens_per_sec, iteration_time_sec, per-rank throughput and iteration times.

Run via the driver (auto-detects workload_card.yaml in the workload directory):
```
python -m tools.main --workload-dir /path/to/workload --tools throughput_group_16
```

Run directly:
```
python - <<'PY'
from tools.throughput_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0", workload_card_path="/path/to/workload_card.yaml"))
PY
```
