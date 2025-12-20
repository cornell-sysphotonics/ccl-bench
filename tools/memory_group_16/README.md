# memory_group_16

Reports peak/reserved/active memory and fragmentation from torch memory snapshots near a trace directory.

- Inputs: iteration directory; the tool searches sibling/parent memory_snapshot folders for *_memory_snapshot.pickle files.
- Outputs: peak reserved/allocated/active memory, fragmentation ratios, allocation stats, per-rank details.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools memory_group_16
```

Run directly:
```
python - <<'PY'
from tools.memory_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
