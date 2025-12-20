# fsdp_group_16

Analyzes FSDP operations (all_gather, reduce_scatter, reshard, copy_in/out, pre/post hooks) from torch profile traces.

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: average FSDP overhead percentage, communication/copy timings, per-operation and per-layer breakdowns, per-rank stats.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools fsdp_group_16
```

Run directly:
```
python - <<'PY'
from tools.fsdp_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
