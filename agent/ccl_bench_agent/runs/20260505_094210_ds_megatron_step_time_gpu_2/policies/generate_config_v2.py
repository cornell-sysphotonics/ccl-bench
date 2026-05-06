
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Adaptive config generator for LLM training workloads.
    
    Key insight: gpu_step_score rewards using fewer GPUs efficiently.
    Score ≈ 0.5*(total_gpus/used_gpus) + 0.5*(reference_time/step_time)
    
    Strategy: Find the minimum GPU count that fits the model in memory,
    then optimize step time within that constraint.
    """
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    num_layers = workload.get("num_layers", None)
    batch_size = workload.get("batch_size", 8)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "")
    
    # Parse config space for valid choices
    config_space = {}
    for dim in workload.get("config_space", []):
        config_space[dim["key"]] = dim["choices"]
    
    tp_choices = sorted(config_space.get("tp", [1, 2, 4]))
    pp_choices = sorted(config_space.get("pp", [1, 3, 9]))
    dp_choices = sorted(config_space.get("dp", [1, 2, 4]))
    ep_choices = sorted(config_space.get("ep", [1, 2, 4]))
    mbs_choices = sorted(config_space.get("micro_batch_size", [1, 2, 4]))
    ac_choices = config_space.get("activation_checkpointing", [True, False])
    
    # Default num_layers based on model family
    if num_layers is None:
        if "deepseek-v2-lite" in model_family.lower():
            num_layers = 27
        elif "deepseek" in model_family.lower():
            num_layers = 60
        elif "llama" in model_family.lower():
            num_layers = 32
        else:
            num_layers = 32
    
    if is_moe:
        # DeepSeek-V2-Lite MoE: 64 experts, 27 layers, ~16B total params
        # Previous results:
        # - tp=4, dp=4, pp=1, ep=4 (16 GPUs) → score 1.007 ✓
        # - tp=2, dp=2, pp=1, ep=2 (4 GPUs) → FAILED (likely OOM)
        
        # Try 8 GPUs: tp=4, dp=2, pp=1, ep=2
        # tp=4 keeps tensor parallelism within a node for good bandwidth
        # dp=2 means 2 data parallel groups
        # ep=2 divides dp=2 and num_experts=64
        # 8 GPUs = 2 nodes, should give ~2x GPU efficiency vs 16 GPUs
        
        tp = 4
        dp = 2
        pp = 1
        ep = 2
        micro_batch_size = 2
        activation_checkpointing = True
        
        # Verify batch divisibility: batch=8, dp=2, mbs=2 → acc_steps=2 ✓
        if batch_size % (dp * micro_batch_size) != 0:
            # Fall back to mbs=1
            micro_batch_size = 1
            if batch_size % (dp * micro_batch_size) != 0:
                # Fall back to dp=1
                dp = 1
                ep = 1
                micro_batch_size = 1
        
        # Verify ep divides dp
        if dp % ep != 0:
            ep = 1
    
    else:
        # Dense models
        # Try to minimize GPUs while fitting model
        tp = min(2, max(tp_choices))
        pp = 1
        dp = 1
        ep = 1
        micro_batch_size = 2
        activation_checkpointing = True
        
        # Ensure batch divisibility
        if batch_size % (dp * micro_batch_size) != 0:
            micro_batch_size = 1
    
    # Validate all choices against config space
    def nearest_valid(val, choices):
        if val in choices:
            return val
        return min(choices, key=lambda x: abs(x - val))
    
    tp = nearest_valid(tp, tp_choices)
    pp = nearest_valid(pp, pp_choices)
    dp = nearest_valid(dp, dp_choices)
    ep = nearest_valid(ep, ep_choices)
    micro_batch_size = nearest_valid(micro_batch_size, mbs_choices)
    
    # Final batch divisibility check
    if batch_size % (dp * micro_batch_size) != 0:
        # Try reducing micro_batch_size
        for mbs in sorted(mbs_choices):
            if batch_size % (dp * mbs) == 0:
                micro_batch_size = mbs
                break
        else:
            micro_batch_size = mbs_choices[0]
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
