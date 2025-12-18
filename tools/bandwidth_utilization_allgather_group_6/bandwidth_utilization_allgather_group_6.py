import yaml
from pathlib import Path

def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization for allgather from the exported sqlite file from nsys.

    Note that AllGather has only been calculated for tp>1 and for the last stage of pp when pp>1.

    n/a for llama tp=1 and node 0 of qwen pp=2

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        Dict[str, float]: The statistics of bandwidth utilization for allgather.
    """
    dir_name = Path(directory).name
    sqlite_path = Path(directory) / "nsys_0.sqlite"
    workload_card_path = Path(directory) / (dir_name + ".yaml")

    # Parse workload card to get metadata
    with open(workload_card_path, 'r') as f:
        workload_card = yaml.safe_load(f)
        model_family = workload_card["workload"]["model"]["model_family"]
        tp = workload_card["Model-executor"]["model_plan_parallelization"]["tp"]

    if model_family == "deepseek-v2-lite":
        pass
    elif model_family == "llama-3.1-8B":
        if tp == 1:
            return "n/a"
    elif model_family == "qwen-32b":
        pass
    else:
        return "n/a"

    ret = {
        "median": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "p25": 0.0,
        "p75": 0.0,
        "p99": 0.0,
    }

    return ret