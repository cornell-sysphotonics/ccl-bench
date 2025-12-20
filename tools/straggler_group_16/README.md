# straggler_group_16

Detects slow ranks and load imbalance from torch profile traces.

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: per-rank durations, kernel/communication time stats, straggler list, coefficient of variation, load-imbalance percentage.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools straggler_group_16
```

Run directly:
```
python - <<'PY'
from tools.straggler_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
