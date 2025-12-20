# Tool Development

Tool development: Byungsoo, Jamal

Metric collection: Byungsoo, Jinkun

## Pipeline

1. Move target trace to `ccl-bench/trace_collection/<trace_name>`

    Example: `ccl-bench/trace_collection/llama3-8B_torchtitan_perlmutter`

2. Define metrics

    Metrics should include a numeric value that can be surfaced in benchmarks. Additional shapes (distributions, time series) are fine when helpful.

3. Develop tools
    ```
    Input: list[nsys_rep], list[kineto_trace], list[pytorch_et_trace]  # stored in trace directory
    Output: float | int | dict
    ```

4. Define tool-trace mapping

    Not all metrics come from the same trace source. Each tool should enforce its own trace checks (for example, minimum GPU count when computing communication bandwidth).

5. Calculate metrics

    ```
    python -m tools.main --workload-dir <trace directory> --tools <tool_name>
    ```

    Each tool directory now has a short README with inputs, outputs, and usage examples.

## Available tools

- [coll_call_num](./coll_call_num): Count NCCL communication kernels from a Kineto trace.
- [communication_group_16](./communication_group_16): Collective operation time and counts from torch profile traces.
- [fsdp_group_16](./fsdp_group_16): FSDP operation overhead, communication, and per-layer timing.
- [gpu_utilization_group_16](./gpu_utilization_group_16): GPU busy/idle percentages from kernel and memcpy spans.
- [kernel_group_16](./kernel_group_16): Top kernels and type breakdown from CUDA kernel events.
- [memory_group_16](./memory_group_16): Peak/reserved/active memory plus fragmentation from memory snapshots.
- [overlap_group_16](./overlap_group_16): Compute/communication overlap efficiency and bubble time.
- [straggler_group_16](./straggler_group_16): Straggler detection and load-imbalance stats across ranks.
- [throughput_group_16](./throughput_group_16): Tokens-per-second throughput using trace timing and workload card.
- [training_phases_group_16](./training_phases_group_16): Forward/backward/optimizer time breakdown from CPU events.
