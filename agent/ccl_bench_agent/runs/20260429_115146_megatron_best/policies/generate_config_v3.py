
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Iteration 3: Try tp=2, pp=1, dp=8, mbs=2, ac=False
    
    History:
    - tp=4, dp=4, pp=1, mbs=2, ac=False -> 8.358 (success)
    - tp=4, dp=4, pp=1, mbs=4, ac=False -> SIGSEGV (OOM)
    - tp=2, dp=8, pp=1, mbs=2, ac=True -> exit 143 (timeout/killed)
    
    Hypothesis: tp=2 reduces TP comm overhead. Previous tp=2 failure had ac=True
    which adds 33% recompute - may have caused timeout. Try without AC.
    If dp=8 cross-node is too slow, we'll fall back to dp=4 variants.
    
    Alternative: try tp=2, pp=2, dp=4 to keep dp=4 (proven) while reducing TP.
    Going with tp=2, pp=2, dp=4, mbs=4, ac=False - this keeps dp=4 and reduces 
    memory per GPU via pp=2, potentially allowing mbs=4.
    
    Actually, let's be safe and try tp=2, pp=2, dp=4, mbs=2, ac=False.
    """
    total_gpus = environment.get("total_gpus", 16)
    gpus_per_node = environment.get("gpus_per_node", 4)
    gpu_memory_gb = environment.get("gpu_memory_gb", 40)
    
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    batch_size = workload.get("batch_size", 32)
    seq_len = workload.get("seq_len", 1024)
    model_family = workload.get("model_family", "").lower()
    precision = workload.get("precision", "bf16")
    
    tp_choices = sorted(valid_choices.get("tp", [1, 2, 4, 8]))
    pp_choices = sorted(valid_choices.get("pp", [1, 2, 4, 8]))
    dp_choices = sorted(valid_choices.get("dp", [1, 2, 4, 8, 16]))
    mbs_choices = sorted(valid_choices.get("micro_batch_size", [1, 2, 4]))
    ac_choices = valid_choices.get("activation_checkpointing", [True, False])
    
    # Estimate model size
    model_size_b = 8.0
    if "70b" in model_family:
        model_size_b = 70.0
    elif "13b" in model_family:
        model_size_b = 13.0
    elif "8b" in model_family:
        model_size_b = 8.0
    elif "7b" in model_family:
        model_size_b = 7.0
    
    # For Llama-3.1-8B on A100-40GB:
    # Model params: 8B, bf16 weights = 16GB, optimizer states ~32GB total
    # With tp=2, pp=2: each GPU holds 1/(2*2)=1/4 of model = ~8GB optimizer states
    # Plus activations for mbs=2, seq=1024 with tp=2: manageable
    
    # Try tp=2, pp=2, dp=4, mbs=2, ac=False
    # This keeps dp=4 (which worked before) and uses pp=2 to reduce memory
    # tp=2 means less TP all-reduce overhead than tp=4
    # pp=2 adds some bubble overhead but for 8 acc_steps (32/(4*2)=4 acc steps) 
    # the bubble fraction is (2-1)/(4+2-1) = 1/5 = 20%
    
    # Actually, let me reconsider. With pp=2, the bubble overhead might hurt.
    # Let me try tp=2, pp=1, dp=8, mbs=2, ac=False instead.
    # The previous tp=2 dp=8 run failed with ac=True. Without AC should be faster.
    
    # Decision: try tp=2, pp=1, dp=8, mbs=2, ac=False
    # dp=8 means 2 DP ranks per node, which requires inter-node communication
    # but with tp=2, each DP rank has half the gradients -> smaller all-reduce
    
    tp = 2
    pp = 1
    dp = total_gpus // (tp * pp)  # 16/2 = 8
    mbs = 2
    ac = False
    
    # Validate
    if dp not in dp_choices:
        # Fallback to proven config
        tp, pp, dp, mbs, ac = 4, 1, 4, 2, False
    
    # Check batch divisibility
    if batch_size % (dp * mbs) != 0:
        mbs = 1
        if batch_size % (dp * mbs) != 0:
            tp, pp, dp, mbs, ac = 4, 1, 4, 2, False
    
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "micro_batch_size": mbs,
        "activation_checkpointing": ac,
    }
