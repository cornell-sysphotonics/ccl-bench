
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Use 8 GPUs instead of 16 to improve gpu_step_score.
    
    Previous results:
    - 16 GPUs (tp=4, dp=4, pp=1, mbs=2, ckpt=False): score 0.5258 ✓
    - 1 GPU (tp=1, dp=1, pp=1, mbs=1, ckpt=True): FAILED (OOM)
    - 4 GPUs (tp=2, dp=2, pp=1, mbs=4, ckpt=True): FAILED (likely OOM - mbs=4 too large)
    
    Try 8 GPUs: tp=4, dp=2, pp=1
    - tp=4 splits model across 4 GPUs (same as run 1, known to work for memory)
    - dp=2 halves the GPU count vs run 1
    - activation_checkpointing=True to be safe on memory
    - micro_batch_size=2 (same as run 1, known to work)
    - 8 GPUs = 2 nodes × 4 GPUs, inter-node comm only for dp allreduce
    
    If score rewards fewer GPUs, using 8 instead of 16 should roughly double the score.
    """
    config = {}
    
    # Parse config space
    config_space = workload.get("config_space", [])
    valid_choices = {}
    for dim in config_space:
        valid_choices[dim["key"]] = dim["choices"]
    
    config["tp"] = 4
    config["dp"] = 2
    config["pp"] = 1
    config["micro_batch_size"] = 2
    config["activation_checkpointing"] = True
    
    return config
