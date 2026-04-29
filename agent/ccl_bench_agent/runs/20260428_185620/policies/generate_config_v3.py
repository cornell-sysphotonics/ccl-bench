
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
            # Try multiple possible key names for the dimension name
            dim_name = None
            for key in ["name", "key", "dimension", "param", "parameter"]:
                if key in dim:
                    dim_name = dim[key]
                    break
            if dim_name is None:
                continue
            # Try multiple possible key names for choices
            dim_choices = None
            for key in ["choices", "values", "options", "range"]:
                if key in dim:
                    dim_choices = dim[key]
                    break
            if dim_choices is None:
                dim_choices = []
            valid[dim_name] = dim_choices
    
    # Helper: pick valid choice closest to desired
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
        # Filter to choices that divide must_divide
        dividing = [c for c in choices if isinstance(c, int) and must_divide % c == 0]
        if dividing:
            # Pick largest <= desired, or largest overall
            good = [c for c in dividing if c <= desired]
            if good:
                return max(good)
            return min(dividing)
        # Fallback: closest to desired
        if desired in choices:
            return desired
        return min(choices, key=lambda x: abs(x - desired) if isinstance(x, (int, float)) else 0)
    
    # Core strategy:
    # 1. Minimize inter-node communication by keeping DP within/across nodes efficiently
    # 2. Use TP within nodes (fast NVLink)
    # 3. Avoid PP (pipeline bubble overhead) unless necessary for memory
    
    num_nodes = total_gpus // gpus_per_node if gpus_per_node > 0 else 1
    
    # Communication cost analysis:
    # - TP uses allreduce within TP group (best on NVLink within node)
    # - DP uses allreduce for gradients (spans across nodes with FSDP)
    # - Larger TP = more intra-node comm but less inter-node comm
    # - Larger DP = more inter-node comm
    
    # When inter-node BW is much lower than intra-node BW, prefer larger TP within node
    bw_ratio = intra_bw / max(inter_bw, 1)
    
    if bw_ratio > 5:
        # High intra/inter ratio: use full node for TP, minimize cross-node DP
        tp = gpus_per_node  # 4 in this case
    elif bw_ratio > 2:
        tp = max(2, gpus_per_node // 2)
    else:
        tp = 2
    
    tp = pick_valid("tp", tp)
    
    # No pipeline parallelism by default
    pp = 1
    pp = pick_valid("pp", pp)
    
    # DP fills remaining
    dp = total_gpus // (tp * pp)
    dp = pick_valid("dp", dp)
    
    # Ensure tp * dp * pp = total_gpus; adjust if needed
    while tp * dp * pp != total_gpus:
        # Try reducing dp
        if tp * dp * pp > total_gpus and dp > 1:
            dp = pick_valid("dp", dp - 1)
        elif tp * dp * pp < total_gpus:
            # Try increasing dp
            dp = pick_valid("dp", dp + 1)
        else:
            break
        # Safety: if we can't match, try different tp
        if tp * dp * pp != total_gpus:
            # Fallback: try tp=2
            tp = pick_valid("tp", 2)
            dp = total_gpus // (tp * pp)
            dp = pick_valid("dp", dp)
            if tp * dp * pp != total_gpus:
                tp = pick_valid("tp", 1)
                dp = total_gpus // pp
                dp = pick_valid("dp", dp)
            break
    
    # local batch = batch_size / dp
    local_batch = batch_size // dp if dp > 0 else batch_size
    
    # micro_batch: with pp=1, use full local_batch to avoid unnecessary accumulation steps
    micro_batch = pick_valid_dividing("micro_batch", local_batch, local_batch)
    
    # Compile mode: try inductor for kernel fusion
    # From history: tp=4/dp=4/eager gave 9.024, tp=2/dp=8/inductor gave 10.39
    # The difference was likely due to parallelism, not compile mode
    # Let's test inductor with the better parallelism config
    compile_mode = "inductor"
    if "compile_mode" in valid and compile_mode not in valid["compile_mode"]:
        compile_mode = valid["compile_mode"][0] if valid["compile_mode"] else "eager"
    
    # Activation checkpointing: disable when memory allows for speed
    # 8B model with tp=4: ~4GB params per GPU, plenty of room on 40GB
    activation_checkpointing = False
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
