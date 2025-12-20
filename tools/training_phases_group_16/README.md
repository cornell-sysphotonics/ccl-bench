# training_phases_group_16

Breaks down training time into forward, backward, optimizer, and FSDP-specific phases using CPU/user_annotation events.

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: average phase times/percentages, per-rank stats, and per-layer FSDP timing when available.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools training_phases_group_16
```

Run directly:
```
python - <<'PY'
from tools.training_phases_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
