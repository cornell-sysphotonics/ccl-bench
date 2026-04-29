
def generate_config(workload: dict, environment: dict) -> dict:
    total_gpus = environment.get("total_gpus", 8)
    gpus_per_node = environment.get("gpus_per_node", 4)
    
    # Try tp=2, dp=4, pp=1 with micro_batch_size=2
    # Less TP communication overhead, more DP parallelism
    # micro_batch_size=2 for better compute efficiency
    tp = 2
    pp = 1
    dp = total_gpus // (tp * pp)  # = 4
    
    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch_size": 2,
        "activation_checkpointing": False,
    }
