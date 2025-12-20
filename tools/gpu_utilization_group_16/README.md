# gpu_utilization_group_16

Calculates GPU busy/idle percentages from kernel and memcpy spans in torch profile traces.

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: average utilization percentages, idle time, kernel/memcpy counts, per-rank stats, and basic device info.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools gpu_utilization_group_16
```

Run directly:
```
python - <<'PY'
from tools.gpu_utilization_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
