
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History analysis (sorted by score, lower is better):
    - tp=4, pp=4, dp=1, mbs=1, no ckpt: score 0.4747 (BEST!)
    - tp=4, pp=2, dp=2, mbs=1, no ckpt: score 3.409
    - tp=2, pp=2, dp=4, mbs=1, no ckpt: score 6.061
    - tp=2, pp=2, dp=4, mbs=2, no ckpt: score 6.524
    - tp=4, dp=4, pp=1, mbs=2, no ckpt: score 8.697
    
    Key lessons:
    - dp=1 is massively better (0.4747 vs 3.409) — eliminating all-reduce is huge
    - tp=4 within node uses fast NVLink
    - pp=4 across nodes uses lightweight point-to-point
    - mbs=1 gives best pipeline efficiency
    - No activation checkpointing when memory allows
    
    New attempt: tp=2, pp=4, dp=2, mbs=1, no ckpt
    - Reduces TP overhead (tp=2 vs tp=4, less intra-node comm)
    - dp=2 all-reduce is very cheap (only 2 ranks)
    - PP=4 maintains efficient pipeline structure
    - Each node: 2 TP groups of size 2, belonging to 2 different pipelines
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
    tp_choices = sorted(valid_choices.get("tp", [1, 2, 4, 8]))
    pp_choices = sorted(valid_choices.get("pp", [1, 2, 4, 8]))
    dp_choices = sorted(valid_choices.get("dp", [1, 2, 4, 8, 16]))
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ckpt_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    num_nodes = total_gpus // gpus_per_node
    
    # Exploration: try tp=2, pp=4, dp=2
    # This reduces TP comm overhead while keeping dp=2 (cheap all-reduce)
    tp = 2
    pp = 4
    dp = total_gpus // (tp * pp)  # 16 / (2*4) = 2
    
    # Validate
    if tp not in tp_choices or pp not in pp_choices or dp not in dp_choices or dp < 1:
        # Fall back to proven best: tp=4, pp=4, dp=1
        tp = 4
        pp = 4
        dp = 1
    
    # Verify product
    if tp * pp * dp != total_gpus:
        tp = 4
        pp = 4
        dp = 1
    
    # Determine micro_batch_size
    samples_per_rank = batch_size // dp if dp > 0 else batch_size
    
    # Use mbs=1 for best pipeline efficiency
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
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": micro_batch,
        "activation_checkpointing": activation_checkpointing,
    }
