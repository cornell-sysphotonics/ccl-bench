
def generate_config(workload: dict, environment: dict) -> dict:
    """Return a configuration dict for the given workload and environment."""
    total_gpus = environment.get("total_gpus", 1)
    gpus_per_node = environment.get("gpus_per_node", 4)
    batch_size = workload.get("batch_size", 1)

    # Build valid choices lookup from config_space
    config_space = workload.get("config_space", [])
    valid = {}
    for dim in config_space:
        valid[dim["name"]] = dim["choices"]

    # Strategy: keep tp within a node, use dp across nodes, pp=1 if possible
    # tp * dp * pp must equal total_gpus
    
    # Try tp=2 (moderate intra-node parallelism), rest as dp
    # For 16 GPUs with 4 per node: tp=2, dp=8, pp=1
    tp = 2
    dp = total_gpus // tp  # 8
    pp = 1
    
    # Validate tp*dp*pp == total_gpus
    if tp * dp * pp != total_gpus:
        # Fallback
        tp = 1
        dp = total_gpus
        pp = 1

    # Ensure values are in valid choices
    if "tp" in valid and tp not in valid["tp"]:
        tp = min(valid["tp"], key=lambda x: abs(x - tp))
    if "dp" in valid and dp not in valid["dp"]:
        dp = min(valid["dp"], key=lambda x: abs(x - dp))
    if "pp" in valid and pp not in valid["pp"]:
        pp = min(valid["pp"], key=lambda x: abs(x - pp))

    # Recalculate if product doesn't match
    if tp * dp * pp != total_gpus:
        # Try all valid combos
        best = None
        for t in valid.get("tp", [1]):
            for d in valid.get("dp", [1]):
                for p in valid.get("pp", [1]):
                    if t * d * p == total_gpus:
                        # Prefer higher dp, lower pp
                        score = d * 100 - p * 10 - t
                        if best is None or score > best[0]:
                            best = (score, t, d, p)
        if best:
            _, tp, dp, pp = best

    # micro_batch: must divide local_batch_size = batch_size // dp
    local_batch = max(1, batch_size // dp)
    
    # Pick largest valid micro_batch that divides local_batch
    mb_choices = valid.get("micro_batch", [1, 2, 4, 8])
    valid_mbs = [m for m in mb_choices if local_batch % m == 0 and m <= local_batch]
    if not valid_mbs:
        valid_mbs = [min(mb_choices)]
    micro_batch = max(valid_mbs)

    # Use activation checkpointing to be safe on memory with A100 40GB
    activation_checkpointing = True

    # Use inductor for potential speedup
    compile_mode = "inductor"

    return {
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "micro_batch": micro_batch,
        "compile_mode": compile_mode,
        "activation_checkpointing": activation_checkpointing,
    }
