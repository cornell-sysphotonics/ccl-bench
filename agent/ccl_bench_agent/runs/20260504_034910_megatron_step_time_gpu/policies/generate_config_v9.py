
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Analysis:
    - Best: tp=4, dp=1, pp=1, mbs=2, ckpt=True → 4 GPUs, score=2.697
    - All 2-GPU configs failed (tp=2/pp=2 both OOM)
    - All 1-GPU configs failed (OOM)
    - tp=4 is the minimum working TP that fits 8B model
    
    Try: tp=4, dp=1, pp=1, mbs=1, ckpt=True → 4 GPUs
    - Same as best config but with mbs=1 instead of mbs=2
    - Smaller micro-batch might reduce memory pressure and could change step timing
    - With global batch=32, dp=1, mbs=1: 32 gradient accumulation steps
    - More accumulation steps but each step is smaller/faster
    - Worth testing to see if gpu_step_score improves
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
    config["micro_batch_size"] = 1
    config["activation_checkpointing"] = True
    
    return config
