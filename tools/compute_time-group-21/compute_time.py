"""
Compute total compute time metric
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


def compute_total_compute_time_s(df: pd.DataFrame) -> float:
    comp_df = df[df["kind"] == "compute"].copy()
    comp_time_s = np.nansum(comp_df["T_s"]) if comp_df["T_s"].notna().any() else comp_df["dur_s"].sum()
    return float(comp_time_s)


def compute_metric(trace_json_path: str) -> float:
    df = prepare_dataframe(trace_json_path)
    return compute_total_compute_time_s(df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_time.py <trace_json_path>", file=sys.stderr)
        sys.exit(1)
    
    trace_path = sys.argv[1]
    result = compute_metric(trace_path)
    print(result)
