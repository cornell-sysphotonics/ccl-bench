"""
Seed configuration policy for the CCL-Bench ADRS agent.

The agent iteratively refines `generate_config`. Each version is a self-contained
Python program: given a workload card and environment, it returns a dict of
configuration key-value pairs.

Execution history is passed only to update_policy (the LLM step), not here.
generate_config is a pure function of workload and environment.
"""


def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment.

    Args:
        workload:    Workload card fields — model_family, phase, batch_size, seq_len,
                     num_heads, num_layers, precision, config_space (list of tunable
                     dimensions with valid choices), run_script, trace_dir, etc.
        environment: Hardware/software descriptor — gpu_model, gpu_memory_gb,
                     total_gpus, gpus_per_node, intra/inter_node_bandwidth_gbps,
                     framework, framework_version.

    Returns:
        dict of configuration key-value pairs matching config_space keys, e.g.:
          {"tp": 4, "dp": 8, "pp": 1, "micro_batch": 4, "compile_mode": "inductor"}
    """
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    batch_size    = workload.get("batch_size", 1)

    # Baseline: fill one node with TP (low-latency intra-node),
    # scale data parallelism across nodes.
    tp = min(gpus_per_node, 8)
    dp = max(1, total_gpus // tp)
    pp = 1
    micro_batch = max(1, batch_size // dp)

    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": "eager",
        "activation_checkpointing": False,
    }
