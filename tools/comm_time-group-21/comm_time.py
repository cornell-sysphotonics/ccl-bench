"""
Compute communication time metrics
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


def compute_total_comm_time_s(df: pd.DataFrame) -> float:
    comm_df = df[df["kind"] == "comm"].copy()
    comm_time_s = np.nansum(comm_df["T_s"]) if comm_df["T_s"].notna().any() else comm_df["dur_s"].sum()
    return float(comm_time_s)


def compute_avg_comm_kernel_time_s(df: pd.DataFrame) -> float:
    comm_df = df[df["kind"] == "comm"].copy()
    avg_comm_kernel_time_s = np.nanmean(comm_df["T_s"]) if comm_df["T_s"].notna().any() else comm_df["dur_s"].mean()
    return float(avg_comm_kernel_time_s)


def compute_allreduce_comm_time_s(df: pd.DataFrame) -> float:
    comm_df = df[df["kind"] == "comm"].copy()
    bw_df = comm_df.dropna(subset=["message_bytes", "T_s"]).copy()
    bw_df = bw_df[(bw_df["message_bytes"] > 0) & (bw_df["T_s"] > 0)]
    ar_df = bw_df[bw_df["name_l"].str.contains("allreduce|all-reduce|all_reduce", regex=True, na=False)].copy()
    
    if len(ar_df) and ar_df["T_s"].notna().any():
        return float(np.nansum(ar_df["T_s"]))
    return float(ar_df["dur_s"].sum()) if len(ar_df) else np.nan


def compute_metric(trace_json_path: str, metric_type: str = "total") -> float:
    df = prepare_dataframe(trace_json_path)
    if metric_type == "total":
        return compute_total_comm_time_s(df)
    elif metric_type == "avg_kernel":
        return compute_avg_comm_kernel_time_s(df)
    elif metric_type == "allreduce":
        return compute_allreduce_comm_time_s(df)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python comm_time.py <trace_json_path> [total|avg_kernel|allreduce]", file=sys.stderr)
        sys.exit(1)
    
    trace_path = sys.argv[1]
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "total"
    result = compute_metric(trace_path, metric_type)
    print(result)

