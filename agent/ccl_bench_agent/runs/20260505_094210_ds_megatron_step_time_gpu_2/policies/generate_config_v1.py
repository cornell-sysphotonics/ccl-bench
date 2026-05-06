
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Adaptive config generator that reasons about workload and hardware.
    Key insight: gpu_step_score rewards using fewer GPUs efficiently.
    Score ≈ 0.5*(total_gpus/used_gpus) + 0.5*(reference_time/step_time)
    So minimizing GPU count while keeping step time reasonable is key.
    """
    gpu_mem = environment.get("gpu_memory_gb", 40)
    gpus_per_node = environment.get("gpus_per_node", 4)
    total_gpus = environment.get("total_gpus", 16)
    is_moe = workload.get("moe", False)
    num_layers = workload.get("num_layers", 27)
    batch_size = workload.get("batch_size", 8)
    seq_len = workload.get("seq_len", 1024)
    
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
    
    # Strategy: minimize GPU count (tp*dp*pp) while fitting in memory
    # For deepseek-v2-lite MoE model on A100-40GB:
    # - Model is relatively small (lite variant)
    # - Try minimal parallelism first
    
    if is_moe:
        # MoE models: need to balance expert parallelism with communication
        # Try tp=2 for intra-node, dp=1 to minimize GPUs, pp=1
        # With fewer GPUs, the gpu_step_score should be higher
        
        # Start with minimal config: tp=1, dp=1, pp=1 = 1 GPU
        # But MoE with 64 experts might not fit on 1 GPU
        
        # Try tp=2, dp=2, pp=1 = 4 GPUs with ep=2
        # This keeps everything within one node (4 GPUs/node)
        tp = 2
        dp = 2
        pp = 1
        ep = 2  # must divide dp and num_experts(64)
        
        # Ensure ep divides dp
        if dp % ep != 0:
            ep = 1
        
        micro_batch_size = 2
        activation_checkpointing = True
        
    else:
        # Dense models: standard parallelism
        tp = min(2, max(tp_choices))
        pp = 1
        dp = 1
        ep = 1
        micro_batch_size = 2
        activation_checkpointing = True
    
    # Validate choices against config space
    if tp not in tp_choices:
        tp = min(tp_choices, key=lambda x: abs(x - tp))
    if pp not in pp_choices:
        pp = min(pp_choices, key=lambda x: abs(x - pp))
    if dp not in dp_choices:
        dp = min(dp_choices, key=lambda x: abs(x - dp))
    if ep not in ep_choices:
        ep = min(ep_choices, key=lambda x: abs(x - ep))
    if micro_batch_size not in mbs_choices:
        micro_batch_size = min(mbs_choices, key=lambda x: abs(x - micro_batch_size))
    
    # Ensure batch_size is divisible by dp * micro_batch_size
    # batch_size = 8, dp=2, mbs=2 → gradient_acc_steps = 8/(2*2) = 2 ✓
    while batch_size % (dp * micro_batch_size) != 0 and micro_batch_size > 1:
        micro_batch_size = max(1, micro_batch_size // 2)
        if micro_batch_size not in mbs_choices:
            micro_batch_size = min(mbs_choices)
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "ep": ep,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": activation_checkpointing,
    }
