# Tool Development

Tool development: Byungsoo, Jamal (wasn't sure we we need to edit)

Metric collection: Byungsoo, Jinkun

## Pipeline

1. Move target trace to `ccl-bench/trace_collection/<trace_name>`

   Example: `ccl-bench/trace_collection/llama-3.18b-fsdp_4-torchprime_xla_tpu-group-21`

   This is usually a zip file, so please extract it before so you can access the files. The usual
   file structure looks like :
   llama-3.18b-fsdp_4-torchprime_xla_tpu-group-21`
   --> Folder (named the date the trace was made)
   ----> Zipped Folder (contains trace.json) -> (Unzip THIS) \*\* This is the file we need to run the script it must be extracted!
   ----> xplane.pb file (this is used elsewhere but not for this script)

2. Define metrics

   Metrics should always include a number (integer, float) that could be presented on the benchmark.
   Other metric formats could be collected in addition, such as distributions or time series.

   Examples of metrics we support:

   - Time-based metrics: wall time, compute time, communication time
   - Utilization metrics: compute utilization proxy, communication fraction
   - Performance metrics: bandwidth, throughput, FLOPs
   - Model parameters: Hockney model alpha/beta for communication modeling

   See the complete list of supported metrics in the "Metrics" section below.

3. Develop tools
   ```
   Input: Chrome trace JSON files (.json or .json.gz) stored in the trace directory
   Output: float | int
   ```
4. Define tool-trace mapping

    Not all the metrics can be derived from one trace, and not all traces can be used to calculate one metric. So a matching checker should be implemented inside every tool to enforce certain matching constraints. An easy example would be checking that the number of GPUs is greater than 1 in the trace by reading the workload card located inside the trace folder when you are calculating network bandwidth utilization, as you need to have multiple GPUs for communication.
5. Calculate metrics


## Metrics 
### Public
1. [Tool ready] `coll_call_num`: number of NCCL communication calls from one GPU in one iteration
17. [Group 4 Tool ready] `mfu`: model flop utilization, representing the efficiency of the model's computation
1. `avg_step_time` (second): average step time (inference: decode, training: forward + backward)

### Private
2.	Group 9 `break_down_steps`: breaks down total GPU kernel time into major components (e.g., communication, attention, MoE routing, MoE expert compute, and other)
3.	Group 9 `communication_ratio`: percentage of communication time over total GPU kernel time
4.	Group 9 `total_communication_time`: total time spent in NCCL communication kernels (e.g., all-reduce/all-to-all/sendrecv)
5.	Group 9 `total_kernel_time`: total GPU kernel execution time across the entire iteration
6.  Group 9 `communication_fraction` - % of time in communication kernels (NCCL, allreduce, etc.)
7.  Group 9 `moe_fraction` - % of time in Mixture of Experts kernels
8.  Group 9 `dominant_kernel_concentration` - % of time in the top kernel (identifies bottlenecks)
9. Group 9 `aggregate_gpu_utilization` - Overall GPU utilization across trace duration
10. Group 9 `mean_sm_coverage` - Average Streaming Multiprocessor coverage
11. Group 9 `memory_transfer_overhead` - % of time spent in memory transfers
12. Group 9 `average_memory_bandwidth` - Average memory bandwidth in GB/s
13. Group 9 `compute_bound_fraction` - % of time in compute-bound kernels
14. Group 9 `memory_bound_fraction` - % of time in memory-bound kernels
15. Group 9 `load_imbalance_ratio` - Max/Min GPU time ratio (for multi-GPU)
16. Group 9 `communication_overlap_ratio` - Ratio of overlapping communication
2. `throughput_tokens_sec`: throughput measured in tokens per second

3. [Group 1 Tool ready] `mfu`: model flop utilization, representing the efficiency of the model's computation

   Examples:

   ```bash
   # Simple metrics (no extra parameters)
   python main.py --trace ./trace_collection/my_trace --metric wall_time_s
   python main.py --trace ./trace_collection/my_trace --metric total_compute_time_s

6. [Group 1 Tool ready] `traffic_window`: time intervals between traffic in different parallelism

   # Throughput metrics (optional model_params)
   python main.py --trace ./trace_collection/my_trace --metric throughput
   python main.py --trace ./trace_collection/my_trace --metric estimated_throughput_tokens_per_s --model_params 7e9
   ```

   Or use scripts:

   ```
   ./scripts/get_<name of metric>.sh
   ```

#Note: not all of these metrics have their own script, they are all calculated in some degree and are included in our original metrics script. This was orginally in a Colab notebook so it would be easier to share and collaborate on the code, so we had to split it up after

## Metrics

Metrics marked with _M_PARAMS_ can optionally use `--model_params` for more accurate throughput estimation.

1. `wall_time_s`: total elapsed wall-clock time covered by the trace

2. `total_compute_time_s`: total time spent in compute operations during the trace

3. `total_comm_time_s`: total time spent in communication operations during the trace

4. `avg_comm_kernel_time_s`: average execution time of a single communication kernel

5. `compute_utilization_proxy`: fraction of total wall-clock time spent in computation

6. `communication_fraction`: fraction of total wall-clock time spent in communication

7. `num_comm_kernels`: number of communication kernels executed in the trace

8. `avg_comm_bandwidth_GBps`: average achieved communication bandwidth across valid kernels (GB/s)

9. `allreduce_comm_time_s`: total time spent in AllReduce communication operations

10. `hockney_alpha_s`: latency term (α) from fitting the AllReduce Hockney communication model

11. `hockney_beta_s_per_byte`: bandwidth cost term (β) from the AllReduce Hockney model

12. `hockney_inverse_beta_Bps`: inverse of β, representing effective communication bandwidth (bytes/sec)

13. `achieved_flops_from_trace_json`: achieved model FLOPs per second estimated from trace events

14. `total_model_flops_from_args`: total model FLOPs summed from trace event arguments

15. `flops_per_token_used` _M_PARAMS_: FLOPs per token value used for throughput estimation (optional `--model_params`)

16. `estimated_total_tokens` _M_PARAMS_: estimated number of tokens processed during the trace (optional `--model_params`)

17. `estimated_throughput_tokens_per_s` _M_PARAMS_: estimated throughput measured in tokens per second (optional `--model_params`)

14. `ttft_group_6`: Extract the median of TTFT in milliseconds from a sglang benchmark JSONL file.

15. `tpot_group_6`: Extract the median of TPOT in milliseconds from a sglang benchmark JSONL file.

16. `throughput_group_6`: Extract the total tokens processed per second from a sglang benchmark JSONL file.

17. `kernel_compute_time_group_6`: Calculate the kernel compute time in seconds from the exported sqlite file from nsys. If there are multiple nodes, the compute time from each node is summed.

18. `bandwidth_utilization_allgather_group_6`: Calculate the median of bandwidth utilization for AllGather from the exported sqlite file from nsys. Note that AllGather has only been calculated for tp > 1 and for the last stage of pp when pp > 1. n/a for llama tp = 1 and node 0 of qwen pp = 2

19. `bandwidth_utilization_allreduce_group_6`: Calculate the median of bandwidth utilization for AllReduce from the exported sqlite file from nsys. n/a for llama tp = 1. For qwen-32b with pp = 2, the metric is calculated by combining data from node 0 and node 1.

20. `bandwidth_utilization_alltoall_group_6`: Calculate the average of non-zero values of bandwidth utilization for AllToAll from the exported sqlite file from nsys, which is the value of "NVLink TX Responses User Data [Throughput %]". Only applicable for deepseek. Not applicable for ep=1.

21. `bandwidth_utilization_peertopeer_group_6`: Calculate the average of non-zero values of bandwidth utilization for PeerToPeer from the exported sqlite file from nsys, which is the value of "NVLink TX Responses User Data [Throughput %]". Only applicable for pp > 1. For qwen model, the value is extracted from "PCIe TX Throughput [Throughput %]" instead. If there are multiple nodes, only output the value of node 0.
14. [Group 1 Tool ready] `comm_kernel_breakdown_tpu_group_4`: a breakdown of the number of calls and time spent on communication kernels

15. [Group 4 tool checked, it calculates memory bw] `estimated_bandwidth`: estimated aggregate bandwidth (GB/s) computed from trace data


17. [Group 4 Tool ready] `mfu`: model flop utilization, representing the efficiency of the model's computation
18. [Group 4 Tool ready] `bandwidth_utilization`: fraction of observed NCCL communication bandwidth relative to the expected hardware bandwidth

15. [Group 4 Tool ready] `communication_overhead`: fraction of total GPU kernel time spent in NCCL communication kernels

...
