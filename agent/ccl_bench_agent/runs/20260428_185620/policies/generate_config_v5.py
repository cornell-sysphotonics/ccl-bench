
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus    = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 8)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    intra_bw      = environment.get("intra_node_bandwidth_gbps", 300)
    inter_bw      = environment.get("inter_node_bandwidth_gbps", 100)
    batch_size    = workload.get("batch_size", 1)
    seq_len       = workload.get("seq_len", 1024)
    precision     = workload.get("precision", "bf16")
    model_family  = workload.get("model_family", "").lower()
    
    # Build lookup of valid choices for each dimension
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        if isinstance(dim, dict):
            dim_name = None
            for key in ["name", "key", "dimension", "param", "parameter"]:
                if key in dim:
                    dim_name = dim[key]
                    break
            if dim_name is None:
                for k, v in dim.items():
                    if isinstance(v, str) and k != "type":
                        dim_name = v
                        break
                if dim_name is None:
                    continue
            dim_choices = None
            for key in ["choices", "values", "options", "range"]:
                if key in dim:
                    dim_choices = dim[key]
                    break
            if dim_choices is None:
                dim_choices = []
            valid[dim_name] = dim_choices
    
    def pick_valid(name, desired):
        if name not in valid or not valid[name]:
            return desired
        choices = valid[name]
        if desired in choices:
            return desired
        return min(choices, key=lambda x: abs(x - desired) if isinstance(x, (int, float)) else 0)
    
    def pick_valid_str(name, desired):
        if name not in valid or not valid[name]:
            return desired
        choices = valid[name]
        if desired in choices:
            return desired
        return choices[0]
    
    def pick_valid_bool(name, desired):
        if name not in valid or not valid[name]:
            return desired
        choices = valid[name]
        if desired in choices:
            return desired
        return choices[0]
    
    num_nodes = total_gpus // gpus_per_node if gpus_per_node > 0 else 1
    bw_ratio = intra_bw / max(inter_bw, 1)
    
    # Execution history for this workload (Llama-8B, 16 GPUs, 4 per node):
    # tp=4, dp=4, pp=1, mb=8, eager, AC=False → 9.024 (BEST)
    # tp=4, dp=4, pp=1, mb=8, eager, AC=True  → 9.08
    # tp=4, dp=4, pp=1, mb=8, inductor, AC=False → 9.088
    # tp=2, dp=8, pp=1, mb=4, inductor, AC=False → 10.39
    #
    # Key insight: tp=gpus_per_node is best (uses fast NVLink), eager > inductor
    # AC adds overhead without benefit here
    #
    # New exploration: try pp=2 to reduce inter-node all-reduce
    # tp=4, dp=2, pp=2: TP within node, PP across 2 nodes, DP across 2 groups
    # This reduces DP all-reduce volume by half (dp=2 vs dp=4)
    # But adds pipeline bubble overhead
    
    # Strategy: try tp=4, dp=2, pp=2 with micro_batch to minimize bubble
    if bw_ratio > 5:
        tp = gpus_per_node
    elif bw_ratio > 2:
        tp = max(2, gpus_per_node // 2)
    else:
        tp = 2
    
    tp = pick_valid("tp", tp)
    
    # Try pp=2 to reduce inter-node DP communication
    # pp=2 with dp=2 means less all-reduce across nodes
    pp = 2
    pp = pick_valid("pp", pp)
    
    dp = total_gpus // (tp * pp)
    dp = pick_valid("dp", dp)
    
    # Verify product matches total_gpus; fall back if not
    if tp * dp * pp != total_gpus:
        # Fallback to known best: tp=gpus_per_node, dp=rest, pp=1
        pp = 1
        pp = pick_valid("pp", pp)
        dp = total_gpus // (tp * pp)
        dp = pick_valid("dp", dp)
        if tp * dp * pp != total_gpus:
            for try_tp in [gpus_per_node, gpus_per_node // 2, 2, 1]:
                try_tp = pick_valid("tp", try_tp)
                try_dp = total_gpus // (try_tp * pp)
                try_dp = pick_valid("dp", try_dp)
                if try_tp * try_dp * pp == total_gpus:
                    tp, dp = try_tp, try_dp
                    break
    
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    # For pp>1, want enough micro-batches to fill pipeline (ideally >= pp stages)
    # micro_batch should divide local_batch
    # With pp=2, dp=2: local_batch=16, want micro_batch=4 → 4 micro-batches (good for 2 stages)
    if pp > 1:
        # Want num_microbatches >= 2*pp for good pipeline efficiency
        target_mb = max(1, local_batch // (2 * pp))
        # Find largest valid micro_batch that divides local_batch and gives enough microbatches
        if "micro_batch" in valid and valid["micro_batch"]:
            candidates = [c for c in valid["micro_batch"] 
                         if isinstance(c, int) and c > 0 and local_batch % c == 0]
            if candidates:
                # Prefer micro_batch that gives >= 2*pp microbatches
                good = [c for c in candidates if local_batch // c >= 2 * pp]
                if good:
                    micro_batch = max(good)  # largest mb that still gives enough microbatches
                else:
                    micro_batch = min(candidates)  # smallest to maximize microbatches
            else:
                micro_batch = pick_valid("micro_batch", target_mb)
        else:
            micro_batch = target_mb
    else:
        # pp=1: micro_batch = local_batch for best throughput
        if "micro_batch" in valid and valid["micro_batch"]:
            candidates = [c for c in valid["micro_batch"] 
                         if isinstance(c, int) and c > 0 and local_batch % c == 0]
            if candidates:
                micro_batch = max(candidates)
            else:
                micro_batch = pick_valid("micro_batch", local_batch)
        else:
            micro_batch = local_batch
    
    compile_mode = pick_valid_str("compile_mode", "eager")
    activation_checkpointing = pick_valid_bool("activation_checkpointing", False)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
