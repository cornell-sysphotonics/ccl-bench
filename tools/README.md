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

## Metrics 

1. [Tool ready] `coll_call_num`: number of NCCL communication calls from one GPU in one iteration
2. `throughput_tokens_sec`: throughput measured in tokens per second

3. `mfu`: model flop utilization, representing the efficiency of the model's computation

4. `sm`: streaming multiprocessor utilization, indicating GPU usage efficiency

5. `bubble_size_pipeline`: size of idle time (bubble) in the pipeline

6. `traffic_window`: time intervals between traffic in different parallelism

7. `traffic_distribution`: distribution of traffic across different parallelization

8. `straggler`: the relative lag of the slowest device or process in a communication group

9. `comm_comp_overlap`: overlap percentage between communication and computation phases

10. `token_to_expert_assignment`: per-device assignment of tokens to experts in a model

11. `iteration_wall_clock_time`: total wall-clock time for one iteration

12. `TTFT`: time to first token in inference

13. `TPOT`: time per output token in inference

---

## `comm_comp_metrics-13` (Group 13)

**Location**: `tools/nsys_analyzer/`

Analyzes communication-computation metrics from nsys traces (`.nsys-rep` / `.sqlite`).

### Metrics Implemented

| Metric | Description | Unit |
|--------|-------------|------|
| `iteration_time-13` | Wall-clock time per training iteration (mean, P50, P99) | ms |
| `comm_comp_overlap-13` | Overlap percentage between communication and computation | % |
| `comm_ratio-13` | Communication time ratio (DP/TP/PP/EP breakdown) | 0-1 |
| `idle_ratio-13` | Idle time ratio | 0-1 |
| `traffic_interval-13` | Time interval between consecutive NCCL calls | ms |
| `kernel_duration-13` | Duration statistics for each NCCL operation type | ms |

### Usage

```bash
# Run all metrics
python3 nsys_analyzer/analyze_trace.py <trace_directory> -o result.json

# Run specific metrics
python3 nsys_analyzer/analyze_trace.py <trace_directory> --metrics iteration_time,overlap
```

### Tool Files Structure

```
|-- nsys_analyzer/
    |
    |-- [comm_comp_metrics-13] Communication-Computation Metrics
    |   |-- analyze_trace.py                # Entry point for all metrics
    |   |-- comm_time_breakdown.py          # Total Time, Comm. Ratio, Idle
    |   |-- comm_compute_overlap.py         # Overlap %
    |   |-- traffic_interval_analyzer.py    # Kernel Duration, Call Interval
    |   |-- iteration_time_analyzer.py      # Iteration Time Mean, P99
    |   |-- phase_window_analyzer.py        # Window time between phases
    |   |-- direct_nsys_analyzer.py         # NCCL call count
    |   |-- accurate_comm_time_analyzer.py  # Accurate comm time (helper)
    |
    |-- [nvlink_bandwidth-13] NVLink Bandwidth Metrics
    |   |-- nvlink_bandwidth_analyzer.py    # NVLink RX/TX throughput %
    |
    |-- [Standalone Tools]
        |-- visualize_kernel_breakdown.py   # Top-10 kernel duration bar chart
```

### Input/Output

- **Input**: `.nsys-rep` or `.sqlite` files
- **Output**: JSON with timing statistics

---

## `nvlink_bandwidth-13` (Group 13)

**Location**: `tools/nsys_analyzer/nvlink_bandwidth_analyzer.py`

Analyzes NVLink bandwidth utilization from nsys traces. **Requires GPU metrics sampling enabled** during trace collection (`nsys profile --gpu-metrics-device=all`).

> **Note**: This metric can only be analyzed for traces collected with GPU metrics enabled. Currently only one trace file supports this analysis.

### Metrics Implemented

| Metric | Description | Unit |
|--------|-------------|------|
| `nvlink_rx_throughput-13` | NVLink receive throughput (mean, P50, P95, max) | % |
| `nvlink_tx_throughput-13` | NVLink transmit throughput (mean, P50, P95, max) | % |
| `nvlink_user_data-13` | User data throughput percentage | % |

### Usage

```bash
# Analyze single trace file
python3 nsys_analyzer/nvlink_bandwidth_analyzer.py <trace.sqlite> -o output.png

# Analyze directory
python3 nsys_analyzer/nvlink_bandwidth_analyzer.py <trace_directory> -o output.png
```

### Output

- **Console**: Per-GPU and system-wide NVLink utilization statistics
- **PNG**: Time-series visualization of NVLink RX/TX throughput

### Requirements

- Trace must be collected with `--gpu-metrics-device=all` flag
- Reads from `GPU_METRICS` table in SQLite (metricId 20-27 for NVLink)
