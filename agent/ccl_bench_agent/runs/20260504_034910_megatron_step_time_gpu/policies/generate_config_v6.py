
def generate_config(workload: dict, environment: dict) -> dict:
    """
    Strategy: Minimize GPU count to maximize gpu_step_score.
    
    Results so far:
    - 16 GPUs (tp=4, dp=4, pp=1, mbs=2, ckpt=False): score 0.5258
    - 8 GPUs (tp=4, dp=2, pp=1, mbs=2, ckpt=True): score 1.039
    - 4 GPUs (tp=4, dp=1, pp=1, mbs=2, ckpt=True): score 2.697
    - 2 GPUs (tp=2, dp=1, pp=1, mbs=1, ckpt=True): FAILED
    - 1 GPU (tp=1, dp=1, pp=1, mbs=1, ckpt=True): FAILED (OOM)
    - 4 GPUs (tp=2, dp=2, pp=1, mbs=4, ckpt=True): FAILED (OOM)
    
    Score scales roughly inversely with GPU count.
    
    tp=2, dp=1, pp=1 (2 GPUs) failed - likely OOM since tp=2 still needs
    full model activations on each GPU.
    
    Try pp=2, tp=1, dp=1 (2 GPUs):
    - PP=2 splits 32 layers into 2 stages of 16 layers each
    - Each GPU holds only half the model weights (~8GB bf16)
    - With activation_checkpointing=True and mbs=1, activation memory is minimal
    - Both GPUs on same node → fast intra-node communication
    - batch=32 with dp=1 → 32 micro-batches in pipeline
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
