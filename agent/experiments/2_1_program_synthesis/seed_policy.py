"""
Parallelism policy function.

This module defines a single function `policy` that maps workload and
environment parameters to a parallelism configuration (tp, dp, pp, micro_batch).

The agent will iteratively refine this function to minimize wall time
across a suite of test workloads.
"""


def policy(
    model: str,
    batch_size: int,
    seq_len: int,
    dmodel: int,
    num_heads: int,
    num_kv_heads: int,
    num_stacks: int,
    precision: str,
    total_gpus: int,
    gpu_memory_gb: int,
    gpus_per_node: int,
    intra_node_bandwidth_gbps: int,
    inter_node_bandwidth_gbps: int
) -> dict:
    """Return {"tp": int, "dp": int, "pp": int, "micro_batch": int} for the given workload.

    Constraints:
        - tp * dp * pp <= total_gpus
        - tp must divide num_heads and num_kv_heads
        - 1 <= micro_batch <= batch_size / (dp * pp)
        - The resulting per-GPU memory must fit in gpu_memory_gb

    This is a simple heuristic baseline. The agent should improve upon it.
    """
    # Baseline heuristic: use tp=4 for memory, fill rest with dp
    tp = 4
    dp = total_gpus // tp
    pp = 1
    micro_batch = batch_size // (dp * pp)

    return {"tp": tp, "dp": dp, "pp": pp, "micro_batch": micro_batch}
