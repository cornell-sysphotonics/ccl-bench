
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History analysis:
    - tp=4, pp=2, dp=2, mbs=1, no ckpt: score 3.409 (BEST)
    - tp=2, pp=2, dp=4, mbs=1, no ckpt: score 6.061
    - tp=2, pp=2, dp=4, mbs=2, no ckpt: score 6.524
    - tp=4, dp=4, pp=1, mbs=2, no ckpt: score 8.697
    
    Key lessons:
    - tp=4 within node (using fast NVLink) is beneficial
    - pp=2 helps, mbs=1 gives best pipeline efficiency
    - dp=2 (less all-reduce) was much better than dp=4
    - No activation checkpointing when memory allows
    
    New attempt: tp=4, pp=4, dp=1, mbs=1, no ckpt
    - Eliminates data-parallel all-reduce entirely (dp=1)
    - pp=4 with 16 microbatches: bubble = 3/19 ≈ 15.8%
    - tp=4 within each node's 4 GPUs
    - pp=4 across the 4 nodes (inter-node, but only point-to-point)
    - Point-to-point PP communication is much lighter than all-reduce
    """
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    inter_bw = environment.get("inter_node_bandwidth_gbps", 25)
    intra_bw = environment.get("intra_node_bandwidth_gbps", 600)
    
    # Extract config space to know valid choices
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    batch_size = workload.get("batch_size", 16)
    seq_len = workload.get("seq_len", 1024)
    precision = workload.get("precision", "bf16")
    model_family = workload.get("model_family", "").lower()
    
    # Get valid choices
    tp_choices = valid_choices.get("tp", [1, 2, 4, 8])
    pp_choices = valid_choices.get("pp", [1, 2, 4, 8])
    dp_choices = valid_choices.get("dp", [1, 2, 4, 8, 16])
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ckpt_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    # Number of nodes
    num_nodes = total_gpus // gpus_per_node
    
    # Strategy: tp within node, pp across nodes, minimize dp
    # Best so far: tp=4, pp=2, dp=2 (score 3.409)
    # Try: tp=4, pp=4, dp=1 — eliminate all-reduce, use pp across all nodes
    
    tp = min(gpus_per_node, max(tp_choices))  # Fill each node with TP
    
    # Try pp = num_nodes to eliminate dp
    pp = num_nodes  # 4 nodes -> pp=4
    dp = total_gpus // (tp * pp)  # Should be 1
    
    # Validate
    if tp not in tp_choices:
        tp = max(c for c in tp_choices if c <= gpus_per_node)
    if pp not in pp_choices:
        pp = max(c for c in pp_choices if c <= num_nodes)
    
    dp = total_gpus // (tp * pp)
    if dp not in dp_choices or dp < 1:
        # Fall back to known best: tp=4, pp=2, dp=2
        tp = 4
        pp = 2
        dp = total_gpus // (tp * pp)
        if dp not in dp_choices:
            dp = min(dp_choices, key=lambda x: abs(x - total_gpus // (tp * pp)))
    
    # Determine micro_batch_size
    samples_per_rank = batch_size // dp if dp > 0 else batch_size
    
    # Use mbs=1 for best pipeline efficiency (more microbatches)
    micro_batch = 1
    if micro_batch not in mbs_choices:
        micro_batch = min(mbs_choices)
    
    # Verify it divides evenly
    if samples_per_rank % micro_batch != 0:
        for mb in mbs_choices:
            if samples_per_rank >= mb and samples_per_rank % mb == 0:
                micro_batch = mb
                break
    
    # No activation checkpointing — faster when memory allows
    # With tp=4 and pp=4, memory per GPU is very small, should be fine
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
