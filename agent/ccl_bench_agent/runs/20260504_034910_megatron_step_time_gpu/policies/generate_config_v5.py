
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Minimize GPU count to maximize gpu_step_score.
    
    Results so far:
    - 16 GPUs (tp=4, dp=4, pp=1, mbs=2, ckpt=False): score 0.5258
    - 8 GPUs (tp=4, dp=2, pp=1, mbs=2, ckpt=True): score 1.039
    - 4 GPUs (tp=4, dp=1, pp=1, mbs=2, ckpt=True): score 2.697
    - 1 GPU (tp=1, dp=1, pp=1, mbs=1, ckpt=True): FAILED (OOM)
    - 4 GPUs (tp=2, dp=2, pp=1, mbs=4, ckpt=True): FAILED (OOM)
    
    Score scales roughly inversely with GPU count.
    Try 2 GPUs: tp=2, dp=1, pp=1
    - tp=2 splits 8B model across 2 GPUs (~4B params each, ~8GB bf16)
    - mbs=1 to minimize activation memory
    - activation_checkpointing=True
    - batch=32 with dp=1 → 32 gradient accumulation steps
    - Both GPUs within one node → fast NVLink
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 2
    config["dp"] = 1
    config["pp"] = 1
    config["micro_batch_size"] = 1
    config["activation_checkpointing"] = True
    
    return config
