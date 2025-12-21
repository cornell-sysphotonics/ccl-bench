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

## Tool Directories

| Directory | Description |
|-----------|-------------|
| [`coll_call_num/`](coll_call_num/README.md) | Count NCCL collective communication calls |
| [`nvlink_usage/`](nvlink_usage/README.md) | NVLink throughput metrics (max, avg, total) |
| [`nvlink_kernel_correlation/`](nvlink_kernel_correlation/README.md) | Correlate NVLink peaks with GPU kernels |
| [`nvlink_plotting/`](nvlink_plotting/README.md) | Generate throughput-over-time plots |
| [`nvlink/`](nvlink/README.md) | NVLink report generation |

## Available Metrics via main.py

| Metric | Tool | Description |
|--------|------|-------------|
| `coll_call_num` | coll_call_num | Number of NCCL communication calls |
| `max_throughput` | nvlink_usage | Maximum NVLink throughput (GB/s) |
| `avg_throughput` | nvlink_usage | Average NVLink throughput (GB/s) |
| `total_communication` | nvlink_usage | Total bytes transferred |
| `nvlink_all` | nvlink_usage | All NVLink throughput metrics |
| `peak_kernels` | nvlink_kernel_correlation | Kernels during peak throughput |

## Usage Examples

```bash
# Collective call count
python main.py --trace /path/to/trace --metric coll_call_num

# NVLink throughput metrics
python main.py --trace /path/to/trace --metric nvlink_all --json
python main.py --trace /path/to/trace --metric max_throughput --link 0 --direction tx

# Kernel correlation
python main.py --trace /path/to/trace --metric peak_kernels --gpu 0 --top 5
```

## All Planned Metrics 

1. [Tool ready] `coll_call_num`: number of NCCL communication calls from one GPU in one iteration
2. [Tool ready] `max_throughput`: maximum NVLink throughput in GB/s
3. [Tool ready] `avg_throughput`: average NVLink throughput in GB/s
4. [Tool ready] `total_communication`: total bytes transferred over NVLink
5. [Tool ready] `peak_kernels`: GPU kernels during peak NVLink throughput
6. `throughput_tokens_sec`: throughput measured in tokens per second
7. `mfu`: model flop utilization, representing the efficiency of the model's computation
8. `sm`: streaming multiprocessor utilization, indicating GPU usage efficiency
9. `bubble_size_pipeline`: size of idle time (bubble) in the pipeline
10. `traffic_window`: time intervals between traffic in different parallelism
11. `traffic_distribution`: distribution of traffic across different parallelization
12. `straggler`: the relative lag of the slowest device or process in a communication group
13. `comm_comp_overlap`: overlap percentage between communication and computation phases
14. `token_to_expert_assignment`: per-device assignment of tokens to experts in a model
15. `iteration_wall_clock_time`: total wall-clock time for one iteration
16. `TTFT`: time to first token in inference
17. `TPOT`: time per output token in inference

...
