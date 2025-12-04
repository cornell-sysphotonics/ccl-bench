# Tool Development

Tool development: Byungsoo, Jamal

Metric collection: Byungsoo, Jinkun

## Pipeline

1. Move target trace to `ccl-bench/trace_collection/<trace_name>`

    Example: `ccl-bench/trace_collection/llama3-8B_torchtitan_perlmutter`

2. Define metrics

    Should always include a number (integer, float) that could be presented on the benchmark.
    Other metric format could be collected in addition, such as distribution, or time series.

    Example: number of communication calls for GPU 0 in one iteration

3. Develop tools
    ```
    Input: list[nsys_rep], list[kineto_trace], list[pytorch_et_trace] # stored in trace directory
    Output: float | int
    ```
4. Define tool-trace mapping

    Not all the metrics can be derived from one trace, and not all traces can be used to calculate one metric. So a matching checker should be implemented inside every tool to enforce certain matching constraints. An easy example would be checking that the number of GPUs is greater than 1 in the trace by reading the workload card located inside the trace folder when you are calculating network bandwidth utilization, as you need to have multiple GPUs for communication.
4. Calculate metrics

    ```
    python main.py --trace=<trace directory> --metric=<name of metric>
    # or use scripts
    ./scripts/get_<name of metric>.sh
    ```

## Usage

### Basic Usage

```bash
cd tools
python main.py --trace <trace_directory> --metric <metric_name>
```

### Available Metrics

List all available metrics:
```bash
python main.py --list-metrics
```

### Profile Mode

The `--profile-mode` option specifies the type of profiling traces:
- `torch`: PyTorch/Kineto profiler traces (default)
- `nsys`: NVIDIA Nsight Systems traces
- `auto`: Automatically detect trace type

```bash
# Explicitly specify torch profiler mode
python main.py --trace <dir> --metric iter_time --profile-mode torch
```

### Examples

```bash
# Calculate number of communication calls
python main.py --trace ../trace_collection/llama-3.1-8b-torchtitan-perlmutter-16 --metric coll_call_num

# Calculate throughput in tokens/sec (returns structured JSON)
python main.py --trace ../trace_collection/llama-3.1-8b-torchtitan-perlmutter-16 --metric throughput_tokens

# Get traffic distribution as JSON
python main.py --trace ../trace_collection/deepseek-v2-lite-torchtitan-perlmutter-16 --metric traffic_distribution --output-json
```

### Batch Processing

Use the helper scripts in `scripts/` to run all metrics at once:

```bash
# Run all metrics for LLaMA
./scripts/run_metrics_llama3.sh

# Run all metrics for DeepSeek
./scripts/run_metrics_deepseek.sh

# Run all metrics for Qwen
./scripts/run_metrics_qwen.sh

# Run all metrics for all workloads
./scripts/run_all_metrics.sh
```

## Metrics

### NVTX Dependency

Some metrics can work without NVTX instrumentation, while others require or benefit from it:

| Metric | NVTX Dependency | Notes |
|--------|-----------------|-------|
| coll_call_num | ❌ None | Works from kernel events only |
| throughput_tokens | ❌ None | Uses config + trace timing |
| iter_time | ❌ None | Uses ProfilerStep# events (no NVTX needed) |
| comm_comp_overlap | ❌ None | Classifies kernels by name pattern |
| pipeline_bubble | ⚠️ Partial | Heuristic mode without NVTX (less accurate) |
| traffic_distribution | ⚠️ Partial | Heuristic mode without NVTX (DP/TP ambiguous) |
| straggler_lag | ❌ None | Compares NCCL kernel times across ranks |

### Implemented Metrics

1. **`coll_call_num`**: Number of NCCL communication calls
   - Source: Kineto trace (`kineto_trace_*.json`)
   - Output: Dict with `total_calls`, `calls_per_rank`, `calls_by_type`
   - Counts: AllReduce, ReduceScatter, AllGather, Broadcast, Reduce, SendRecv, AllToAll
   - NVTX: Not required

2. **`throughput_tokens`**: Throughput measured in tokens per second
   - Source: Workload card + Kineto/Torch ET trace timing
   - Output: Dict with `throughput_tokens_per_s`, `total_tokens`, `measured_duration_s`, etc.
   - Formula: `(batch_size × seq_len × iterations) / total_time`
   - NVTX: Not required (uses ProfilerStep# for iteration detection)

3. **`iter_time`**: Average iteration wall-clock time
   - Source: Kineto trace (ProfilerStep# markers)
   - Output: Dict with `avg_iter_time_ms`, `min_iter_time_ms`, `max_iter_time_ms`, `std_iter_time_ms`
   - Uses ProfilerStep# events (automatically added by PyTorch Profiler, no NVTX needed)
   - NVTX: Not required

4. **`comm_comp_overlap`**: Overlap ratio between communication and computation phases
   - Source: Kineto trace (kernel events)
   - Output: Dict with `comm_time_ms`, `comp_time_ms`, `overlap_time_ms`, `overlap_ratio_of_comm`, `overlap_ratio_of_comp`
   - Classifies kernels by name: NCCL = comm, matmul/gemm/attention = comp
   - NVTX: Not required

5. **`pipeline_bubble`**: Size of idle time (bubble) in pipeline parallelism
   - Source: Kineto trace (kernel timelines)
   - Output: Dict with `bubble_ratio`, `bubble_time_ms`, `total_time_ms`, `method`
   - NVTX: Partial (heuristic mode uses gaps between kernels, NVTX mode uses stage markers)
   - Warning: Heuristic mode may be inaccurate without proper stage tagging

6. **`traffic_distribution`**: Distribution of traffic across different parallelization types
   - Source: Kineto trace (NVTX ranges + kernel names)
   - Output: Dict with `dp_bytes`, `tp_bytes`, `pp_bytes`, `ep_bytes`, `fractions`, `mode`
   - Categories: DP, TP, PP, EP, unknown
   - NVTX: Partial (uses heuristics without NVTX, PP/EP are reliable, DP/TP may be ambiguous)

7. **`straggler_lag`**: Relative lag of the slowest device in a communication group
   - Source: Multi-rank Kineto traces
   - Output: Dict with `mean_lag_us`, `p50_lag_us`, `p95_lag_us`, `max_lag_us`, `num_collectives`
   - Groups NCCL kernels by position (k-th kernel) across ranks
   - NVTX: Not required

### Planned Metrics

8. `mfu`: Model FLOP utilization (GPU efficiency)

9. `sm`: Streaming multiprocessor utilization

10. `traffic_window`: Time intervals between traffic in different parallelism

11. `token_to_expert_assignment`: Per-device token to expert assignment (MoE models)

12. `TTFT`: Time to first token (inference)

13. `TPOT`: Time per output token (inference)

## Directory Structure

```
tools/
├── main.py                      # Main entry point
├── README.md                    # This file
├── coll_call_num/
│   ├── __init__.py
│   └── coll_call_num.py        # Communication call counter
├── throughput_tokens/
│   ├── __init__.py
│   └── throughput_tokens.py    # Throughput calculator
├── iter_time/
│   ├── __init__.py
│   └── iter_time.py            # Iteration time calculator
├── comm_comp_overlap/
│   ├── __init__.py
│   └── comm_comp_overlap.py    # Comm/comp overlap analyzer
├── pipeline_bubble/
│   ├── __init__.py
│   └── pipeline_bubble.py      # Pipeline bubble analyzer
├── traffic_distribution/
│   ├── __init__.py
│   └── traffic_distribution.py # Traffic classifier by parallelism
└── straggler_lag/
    ├── __init__.py
    └── straggler_lag.py        # Straggler detection
```

## Adding New Metrics

To add a new metric:

1. Create a new directory under `tools/` with your metric name
2. Create `__init__.py` that exports `metric_cal`
3. Create `<metric_name>.py` with:
   ```python
   from typing import Any

   def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
       """
       Calculate the metric from trace data.

       Args:
           directory: Path to trace directory
           profile_mode: Profile mode - 'torch', 'nsys', or 'auto'

       Returns:
           Dictionary with metric values and metadata
       """
       # Implementation
       return {
           "metric_value": ...,
           "units": "...",
           # ... other relevant fields
       }
   ```
4. Register the metric in `_METRIC_MODULES` dict in `main.py`

## Trace Requirements

Each metric tool expects specific trace files in the trace directory:

| Metric | Required Traces |
|--------|-----------------|
| coll_call_num | `kineto_trace_*.json` |
| throughput_tokens | `kineto_trace_*.json` or `torch_et_*.json`, `workload_card.yaml` |
| iter_time | `kineto_trace_*.json` |
| comm_comp_overlap | `kineto_trace_*.json` |
| pipeline_bubble | `kineto_trace_*.json` (multi-rank preferred) |
| traffic_distribution | `kineto_trace_*.json` |
| straggler_lag | `kineto_trace_*.json` (multi-rank required) |

## Dependencies

- Python 3.8+
- PyYAML (for workload card parsing)

Install dependencies:
```bash
pip install pyyaml
```
