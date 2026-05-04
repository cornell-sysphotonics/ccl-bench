
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Minimize GPU count to maximize gpu_step_score.
    
    Results so far:
    - 16 GPUs (tp=4, dp=4, pp=1, mbs=2, ckpt=False): score 0.5258
    - 8 GPUs (tp=4, dp=2, pp=1, mbs=2, ckpt=True): score 1.039
    - 1 GPU: FAILED (OOM)
    - 4 GPUs (tp=2, dp=2, mbs=4, ckpt=True): FAILED (OOM)
    
    Pattern: Score roughly doubles when halving GPU count.
    
    Try 4 GPUs: tp=4, dp=1, pp=1
    - tp=4 splits model across 4 GPUs (proven to work for memory)
    - dp=1 means no data parallelism (fewer GPUs)
    - All 4 GPUs within one node → fast NVLink communication
    - mbs=2, ckpt=True (known working combo)
    - batch=32 with dp=1 → 32/2=16 gradient accumulation steps
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
    config["micro_batch_size"] = 2
    config["activation_checkpointing"] = True
    
    return config
