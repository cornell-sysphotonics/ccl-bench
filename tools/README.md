# Metric Tools

`tools/` contains the metric implementations used by CCL-Bench. A metric consumes a trace directory and returns a scalar value that can be displayed in the website table.

## Running A Metric

From the repository root:

```bash
python tools/main.py --trace /path/to/trace_dir --metric avg_step_time
```

The trace directory should contain the workload card and any profiler artifacts needed by the selected metric, such as Kineto JSON, XLA trace JSON, Nsight Systems SQLite exports, or benchmark latency JSON files.

## Metric Interface

Each metric lives in its own subdirectory and exposes a callable function used by `tools/main.py`, usually:

```python
def metric_cal(trace_dir: str) -> float:
    ...
```

Metric implementations should:

- validate that the trace directory contains the artifacts they need;
- return `None` when a metric is not applicable to a trace;
- avoid modifying trace artifacts;
- document which profiler format they expect;
- keep units explicit in the README or code comments.

If a metric should appear on the website, add it to `website/benchmark_config.json` and regenerate `website/benchmark_data.json` plus `website/data.js`.

## Dashboard Metrics

The current public dashboard is driven by the metrics configured in `website/benchmark_config.json`. Commonly used metrics include:

| Metric | Meaning |
| --- | --- |
| `avg_step_time` | Average per-step or per-engine-iteration wall time from trace annotations. |
| `ttft` | Trace-derived time to first token for inference rows when available. |
| `tpot` | Trace-derived time per output token for inference rows when available. |
| `mfu` | Model FLOP utilization computed from workload-card model metadata and measured time. |
| `mean_sm_coverage` | Average streaming multiprocessor coverage from GPU kernel trace data. |
| `dominant_kernel_concentration` | Fraction of kernel time spent in the most dominant kernel. |
| `moe_fraction` | Fraction of kernel time attributed to MoE-related kernels. |
| `average_memory_bandwidth` | Average explicit memory transfer bandwidth when trace data exposes bytes and duration. |
| `memory_transfer_overhead` | Fraction of time spent in explicit memory transfers. |
| `communication_fraction` | Fraction of traced time spent in communication kernels or communication events. |
| `total_communication_time` | Total traced communication time. |
| `compute_comm_overlap` | Estimated overlap between compute and communication intervals. |
| `bandwidth_utilization_allgather_group_6` | Estimated AllGather bandwidth utilization for applicable GPU traces. |
| `bandwidth_utilization_allreduce_group_6` | Estimated AllReduce bandwidth utilization for applicable GPU traces. |
| `bandwidth_utilization_alltoall_group_6` | Estimated AllToAll bandwidth utilization for applicable GPU traces. |
| `bandwidth_utilization_reducescatter_group_6` | Estimated ReduceScatter bandwidth utilization for applicable GPU traces. |
| `bandwidth_utilization_peertopeer_group_6` | Estimated point-to-point bandwidth utilization for applicable GPU traces. |

Some metric names still include historical group suffixes. They are kept to avoid breaking existing website configuration and prior benchmark rows.

## Trace Compatibility

Not every metric applies to every trace:

- Kineto or XLA JSON traces are typically used for step-time style metrics.
- Nsight Systems SQLite exports are used for GPU kernel and communication breakdowns.
- vLLM or SGLang benchmark JSON files may provide request-level latency fields.
- Workload cards provide model architecture, phase, parallelism, and hardware metadata.

Metric tools should fail clearly when an expected artifact is missing and should return `None` for intentionally unsupported trace types.

## Adding A Metric

1. Create `tools/<metric_name>/`.
2. Implement the metric function and any small helper functions.
3. Add a short README or module docstring describing required trace artifacts and units.
4. Register the metric in `tools/main.py`.
5. Add a small fixture or documented manual check when possible.
6. Add the metric to `website/benchmark_config.json` only if it should be shown on the public leaderboard.

Keep raw traces and generated profiler dumps out of git unless they are tiny fixtures created specifically for testing.
