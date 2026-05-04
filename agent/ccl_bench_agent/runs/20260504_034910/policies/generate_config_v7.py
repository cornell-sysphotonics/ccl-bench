
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Best so far: tp=4, dp=1, pp=1, mbs=2, ckpt=True → 4 GPUs, score=2.697
    
    Try: tp=4, dp=1, pp=1, mbs=4, ckpt=True → 4 GPUs
    - Larger micro-batch might improve GPU utilization/throughput
    - With dp=1, global_batch=32 → 32/4=8 accumulation steps (vs 16 with mbs=2)
    - Fewer gradient accumulation steps = less overhead
    - tp=4 within single node = fast NVLink communication
    - activation_checkpointing=True to manage memory
    
    If mbs=4 causes OOM, we'll fall back to exploring other 4-GPU configs.
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 4
    config["dp"] = 1
    config["pp"] = 1
    config["micro_batch_size"] = 4
    config["activation_checkpointing"] = True
    
    return config
