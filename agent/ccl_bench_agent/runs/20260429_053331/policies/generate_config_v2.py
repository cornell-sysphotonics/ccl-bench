
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 8)
    
    # Build valid choices lookup
    config_space = workload.get("config_space", [])
    choices = {}
    for dim in config_space:
        choices[dim["key"]] = dim["choices"]
    
    # tp=4, dp=2, pp=1 worked (score 5.889)
    # tp=2, dp=4, pp=1 failed (possibly TP across nodes issue)
    # Try tp=4, dp=2, pp=1 with micro_batch=2 for better compute efficiency
    # global_batch=8, dp=2 → per_dp_batch=4, micro_batch=2 → 2 accumulation steps
    
    tp = 4
    pp = 1
    dp = total_gpus // (tp * pp)  # = 2
    
    # Validate dp
    if dp not in choices.get("dp", [1, 2, 4, 8]):
        dp = 2
    
    micro_batch_size = 2
    
    # Ensure micro_batch_size divides the per-dp batch
    per_dp_batch = batch_size // dp
    if per_dp_batch % micro_batch_size != 0:
        micro_batch_size = 1
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": micro_batch_size,
        "activation_checkpointing": False,
    }
