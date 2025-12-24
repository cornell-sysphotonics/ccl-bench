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

3. [Tool ready] `mfu`: model flop utilization, representing the efficiency of the model's computation

4. `sm`: streaming multiprocessor utilization, indicating GPU usage efficiency

5. `bubble_size_pipeline`: size of idle time (bubble) in the pipeline

6. [Tool ready] `traffic_window`: time intervals between traffic in different parallelism

7. `traffic_distribution`: distribution of traffic across different parallelization

8. `straggler`: the relative lag of the slowest device or process in a communication group

9. `comm_comp_overlap`: overlap percentage between communication and computation phases

10. `token_to_expert_assignment`: per-device assignment of tokens to experts in a model

11. `iteration_wall_clock_time`: total wall-clock time for one iteration

12. `TTFT`: time to first token in inference

13. `TPOT`: time per output token in inference

14. [Tool ready] `bandwidth_utilization`: fraction of observed NCCL communication bandwidth relative to the expected hardware bandwidth

15. [Tool ready] `communication_overhead`: fraction of total GPU kernel time spent in NCCL communication kernels

...
