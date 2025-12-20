# communication_group_16

Breaks down collective communication from torch profile traces (rank*_trace.json).

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: total communication time/count, per-operation breakdown, per-rank statistics.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools communication_group_16
```

Run directly:
```
python - <<'PY'
from tools.communication_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
