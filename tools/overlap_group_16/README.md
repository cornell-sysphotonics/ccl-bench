# overlap_group_16

Measures compute vs. communication overlap using CUDA kernel intervals.

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: overlap efficiency, compute/communication/bubble time, kernel counts, per-rank stats.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools overlap_group_16
```

Run directly:
```
python - <<'PY'
from tools.overlap_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
