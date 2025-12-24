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

1. [Group 1 Tool ready] `coll_call_num`: number of NCCL communication calls from one GPU in one iteration
2. `throughput_tokens_sec`: throughput measured in tokens per second

3. [Group 1 Tool ready] `mfu`: model flop utilization, representing the efficiency of the model's computation

4. `sm`: streaming multiprocessor utilization, indicating GPU usage efficiency

5. `bubble_size_pipeline`: size of idle time (bubble) in the pipeline

6. [Group 1 Tool ready] `traffic_window`: time intervals between traffic in different parallelism

7. `traffic_distribution`: distribution of traffic across different parallelization

8. `straggler`: the relative lag of the slowest device or process in a communication group

9. `comm_comp_overlap`: overlap percentage between communication and computation phases

10. `token_to_expert_assignment`: per-device assignment of tokens to experts in a model

11. `iteration_wall_clock_time`: total wall-clock time for one iteration

12. `TTFT`: time to first token in inference

13. `TPOT`: time per output token in inference

14. `ttft_group_6`: Extract the median of TTFT in milliseconds from a sglang benchmark JSONL file.

15. `tpot_group_6`: Extract the median of TPOT in milliseconds from a sglang benchmark JSONL file.

16. `throughput_group_6`: Extract the total tokens processed per second from a sglang benchmark JSONL file.

17. `kernel_compute_time_group_6`: Calculate the kernel compute time in seconds from the exported sqlite file from nsys. If there are multiple nodes, the compute time from each node is summed.

18. `bandwidth_utilization_allgather_group_6`: Calculate the median of bandwidth utilization for AllGather from the exported sqlite file from nsys. Note that AllGather has only been calculated for tp > 1 and for the last stage of pp when pp > 1. n/a for llama tp = 1 and node 0 of qwen pp = 2

19. `bandwidth_utilization_allreduce_group_6`: Calculate the median of bandwidth utilization for AllReduce from the exported sqlite file from nsys. n/a for llama tp = 1. For qwen-32b with pp = 2, the metric is calculated by combining data from node 0 and node 1.

20. `bandwidth_utilization_alltoall_group_6`: Calculate the average of non-zero values of bandwidth utilization for AllToAll from the exported sqlite file from nsys, which is the value of "NVLink TX Responses User Data [Throughput %]". Only applicable for deepseek. Not applicable for ep=1.

21. `bandwidth_utilization_peertopeer_group_6`: Calculate the average of non-zero values of bandwidth utilization for PeerToPeer from the exported sqlite file from nsys, which is the value of "NVLink TX Responses User Data [Throughput %]". Only applicable for pp > 1. For qwen model, the value is extracted from "PCIe TX Throughput [Throughput %]" instead. If there are multiple nodes, only output the value of node 0.
14. [Group 1 Tool ready] `comm_kernel_breakdown_tpu_group_4`: a breakdown of the number of calls and time spent on communication kernels
15. [Group 1 Tool ready] `estimated_bandwidth`: estimated aggregate bandwidth (GB/s) computed from trace data


17. [Group 4 Tool ready] `mfu`: model flop utilization, representing the efficiency of the model's computation
18. [Group 4 Tool ready] `bandwidth_utilization`: fraction of observed NCCL communication bandwidth relative to the expected hardware bandwidth

15. [Group 4 Tool ready] `communication_overhead`: fraction of total GPU kernel time spent in NCCL communication kernels

...
