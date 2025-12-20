# kernel_group_16

Summarizes CUDA kernel activity, highlighting top kernels and type breakdowns (GEMM, attention, elementwise, memory, NCCL, etc.).

- Inputs: iteration directory with torch Chrome trace JSON files.
- Outputs: top kernels by time, launch counts, per-type summaries, per-rank stats.

Run via the driver:
```
python -m tools.main --workload-dir /path/to/workload --tools kernel_group_16
```

Run directly:
```
python - <<'PY'
from tools.kernel_group_16.metric import metric_cal
print(metric_cal("/path/to/profile_traces/iteration_0"))
PY
```
