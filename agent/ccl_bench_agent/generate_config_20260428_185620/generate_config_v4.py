
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
    
    # Build lookup of valid choices for each dimension - handle various key formats
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
                # Try first string value as name
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
    
    def pick_valid_dividing(name, desired, must_divide):
        """Pick largest valid choice <= desired that divides must_divide."""
        if name not in valid or not valid[name]:
            return desired
        choices = valid[name]
        dividing = [c for c in choices if isinstance(c, int) and must_divide % c == 0]
        if dividing:
            good = [c for c in dividing if c <= desired]
            if good:
                return max(good)
            return min(dividing)
        if desired in choices:
            return desired
        return min(choices, key=lambda x: abs(x - desired) if isinstance(x, (int, float)) else 0)
    
    num_nodes = total_gpus // gpus_per_node if gpus_per_node > 0 else 1
    bw_ratio = intra_bw / max(inter_bw, 1)
    
    # Strategy based on execution history:
    # tp=4/dp=4/eager/no-AC = 9.024 (best)
    # tp=4/dp=4/inductor/no-AC = 9.088
    # tp=2/dp=8/inductor/no-AC = 10.39
    # 
    # Key: tp=gpus_per_node gives best results when intra >> inter BW
    # eager slightly beats inductor
    # Now test: activation_checkpointing=True with best config
    
    # Use TP = gpus_per_node when high BW ratio
    if bw_ratio > 5:
        tp = gpus_per_node
    elif bw_ratio > 2:
        tp = max(2, gpus_per_node // 2)
    else:
        tp = 2
    
    tp = pick_valid("tp", tp)
    pp = pick_valid("pp", 1)
    
    dp = total_gpus // (tp * pp)
    dp = pick_valid("dp", dp)
    
    # Adjust if product doesn't match
    if tp * dp * pp != total_gpus:
        # Try fallbacks
        for try_tp in [gpus_per_node, gpus_per_node // 2, 2, 1]:
            try_tp = pick_valid("tp", try_tp)
            try_dp = total_gpus // (try_tp * pp)
            try_dp = pick_valid("dp", try_dp)
            if try_tp * try_dp * pp == total_gpus:
                tp, dp = try_tp, try_dp
                break
    
    local_batch = batch_size // dp if dp > 0 else batch_size
    micro_batch = pick_valid_dividing("micro_batch", local_batch, local_batch)
    
    # Based on history: eager slightly better than inductor for this workload
    compile_mode = "eager"
    if "compile_mode" in valid and compile_mode not in valid["compile_mode"]:
        compile_mode = valid["compile_mode"][0] if valid["compile_mode"] else "eager"
    
    # Test activation checkpointing - it reduces memory but adds recomputation
    # For 8B model on 40GB A100 with tp=4, memory shouldn't be tight
    # But AC can sometimes help by reducing memory pressure / enabling larger effective batch
    # Testing True this iteration to see the effect
    activation_checkpointing = True
    if "activation_checkpointing" in valid and activation_checkpointing not in valid["activation_checkpointing"]:
        activation_checkpointing = False
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
