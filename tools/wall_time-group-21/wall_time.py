"""
Compute wall time metric
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


def compute_wall_time_s(df: pd.DataFrame) -> float:
    t0, t1 = df["start_s"].min(), df["end_s"].max()
    return (t1 - t0) if np.isfinite(t0) and np.isfinite(t1) else np.nan


def compute_metric(trace_json_path: str) -> float:
    df = prepare_dataframe(trace_json_path)
    return compute_wall_time_s(df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wall_time.py <trace_json_path>", file=sys.stderr)
        sys.exit(1)
    
    trace_path = sys.argv[1]
    result = compute_metric(trace_path)
    print(result)
