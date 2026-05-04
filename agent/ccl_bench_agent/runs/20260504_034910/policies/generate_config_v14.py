
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Best config: tp=4, dp=1, pp=1, mbs=2, checkpointing=False → 2.76
    
    Try tp=2, dp=2, pp=1, mbs=2, checkpointing=True (4 GPUs, all intra-node)
    Previous tp=2,dp=2 with mbs=4 failed. mbs=2 uses less memory, may work.
    If it works with 4 GPUs like tp=4,dp=1, the score could be similar or better.
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 2
    config["dp"] = 2
    config["pp"] = 1
    config["micro_batch_size"] = 2
    config["activation_checkpointing"] = True
    
    return config
