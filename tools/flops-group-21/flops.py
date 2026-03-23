"""
Compute FLOPs metrics
"""
import sys
import os
import importlib.util
import numpy as np
import pandas as pd

# Import utils-group-21
tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(tools_dir, "utils-group-21.py")
spec = importlib.util.spec_from_file_location("utils_group_21", utils_path)
utils_group_21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_group_21)
prepare_dataframe = utils_group_21.prepare_dataframe
safe_float = utils_group_21.safe_float


def sum_model_flops_from_args(df: pd.DataFrame) -> float:
    total = 0.0
    if "args" not in df.columns:
        return 0.0
    for a in df["args"].values:
        if isinstance(a, dict) and "model_flops" in a:
            total += safe_float(a.get("model_flops"), default=0.0)
    return float(total)


def compute_achieved_flops(df: pd.DataFrame) -> float:
    ops = df.dropna(subset=["model_flops", "T_s"]).copy()
    ops = ops[(ops["model_flops"] > 0) & (ops["T_s"] > 0)]
    if len(ops) and ops["T_s"].sum() > 0:
        return float(ops["model_flops"].sum() / ops["T_s"].sum())
    return np.nan


def compute_total_model_flops(trace_json_path: str) -> float:
    df = prepare_dataframe(trace_json_path)
    return sum_model_flops_from_args(df)


def compute_achieved_flops_from_trace(trace_json_path: str) -> float:
    df = prepare_dataframe(trace_json_path)
    return compute_achieved_flops(df)


def compute_metric(trace_json_path: str, metric_type: str = "achieved_flops") -> float: 
    if metric_type == "achieved_flops":
        return compute_achieved_flops_from_trace(trace_json_path)
    elif metric_type == "total_model_flops":
        return compute_total_model_flops(trace_json_path)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flops.py <trace_json_path> [achieved_flops|total_model_flops]", file=sys.stderr)
        sys.exit(1)
    
    trace_path = sys.argv[1]
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "achieved_flops"
    result = compute_metric(trace_path, metric_type)
    print(result)

