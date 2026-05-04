
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Analysis of history:
    - Fewer GPUs = higher score (16→0.53, 8→1.04, 4→2.70)
    - tp=4 works, tp=2 alone fails, tp=1 alone fails (OOM)
    - Best: tp=4, dp=1, pp=1, mbs=2, ckpt=True → 4 GPUs, score=2.697
    - tp=4, mbs=4 failed (OOM)
    
    Try: tp=1, dp=1, pp=2, mbs=1, ckpt=True → 2 GPUs
    - pp=2 splits model layers across 2 GPUs (each has ~half the 8B params)
    - ~4B params per GPU in bf16 = ~8GB weights
    - With ckpt=True and mbs=1, should fit in 40GB A100
    - If works, score could be ~5.4 (2x the 4-GPU score)
    - No tensor parallelism overhead
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 1
    config["dp"] = 1
    config["pp"] = 2
    config["micro_batch_size"] = 1
    config["activation_checkpointing"] = True
    
    return config
