
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Optimized config for LLM training on multi-node GPU clusters.
    
    History analysis (sorted by score, lower is better):
    - tp=4, pp=4, dp=1, mbs=1, no ckpt: score 0.4747 (BEST)
    - tp=4, pp=2, dp=2, mbs=1, no ckpt: score 3.409
    - tp=2, pp=2, dp=4, mbs=1, no ckpt: score 6.061
    - tp=2, pp=2, dp=4, mbs=2, no ckpt: score 6.524
    - tp=4, dp=4, pp=1, mbs=2, no ckpt: score 8.697
    - tp=2, pp=4, dp=2, mbs=1, no ckpt: FAILED
    
    Key lessons:
    - dp=1 is massively better — eliminates all-reduce overhead
    - tp=4 within node uses fast NVLink (600 GB/s)
    - pp=4 across nodes uses lightweight point-to-point
    - mbs=1 maximizes pipeline efficiency (bubble ratio = (pp-1)/num_microbatches)
    
    Exploration: tp=8, pp=2, dp=1, mbs=1
    - tp=8 spans 2 nodes (requires inter-node TP comm — expensive)
    - BUT pp=2 means only 2 pipeline stages → bubble = 1/16 = 6.25%
    - vs current best pp=4 → bubble = 3/16 = 18.75%
    - Question: does 12.5% bubble reduction offset inter-node TP cost?
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
    
    # Try tp=8, pp=2, dp=1 to test if reduced pipeline bubble
    # offsets inter-node TP communication cost
    tp = 8
    pp = 2
    dp = total_gpus // (tp * pp)  # 16 / (8*2) = 1
    
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
