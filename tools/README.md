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
5. Calculate metrics

    ```
    python main.py --trace=<trace directory> --metric=<name of metric>
    # or use scripts
    ./scripts/get_<name of metric>.sh
    ```

## Metrics 

1. [Tool ready] `coll_call_num`: number of NCCL communication calls from one GPU in one iteration
2.	`break_down_steps`: breaks down total GPU kernel time into major components (e.g., communication, attention, MoE routing, MoE expert compute, and other)
3.	`communication_ratio`: percentage of communication time over total GPU kernel time
4.	`total_communication_time`: total time spent in NCCL communication kernels (e.g., all-reduce/all-to-all/sendrecv)
5.	`total_kernel_time`: total GPU kernel execution time across the entire iteration
6. `communication_fraction` - % of time in communication kernels (NCCL, allreduce, etc.)
7. `moe_fraction` - % of time in Mixture of Experts kernels
8. `dominant_kernel_concentration` - % of time in the top kernel (identifies bottlenecks)
9. `aggregate_gpu_utilization` - Overall GPU utilization across trace duration
10. `mean_sm_coverage` - Average Streaming Multiprocessor coverage
11. `memory_transfer_overhead` - % of time spent in memory transfers
12. `average_memory_bandwidth` - Average memory bandwidth in GB/s
13. `compute_bound_fraction` - % of time in compute-bound kernels
14. `memory_bound_fraction` - % of time in memory-bound kernels
15. `load_imbalance_ratio` - Max/Min GPU time ratio (for multi-GPU)
16. `communication_overlap_ratio` - Ratio of overlapping communication