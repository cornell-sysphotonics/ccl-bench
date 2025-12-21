"""
Compute bandwidth metrics
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


def compute_comm_bandwidth_table(comm_df: pd.DataFrame, n_chips: int) -> tuple[pd.DataFrame, float]:
    bw_df = comm_df.dropna(subset=["message_bytes", "T_s"]).copy()
    bw_df = bw_df[(bw_df["message_bytes"] > 0) & (bw_df["T_s"] > 0)]
    bw_df["bandwidth_Bps"] = (n_chips - 1) * bw_df["message_bytes"] / bw_df["T_s"]
    bw_df["bandwidth_GBps"] = bw_df["bandwidth_Bps"] / 1e9
    avg_bandwidth_GBps = bw_df["bandwidth_GBps"].mean() if len(bw_df) else np.nan
    return bw_df, float(avg_bandwidth_GBps) if np.isfinite(avg_bandwidth_GBps) else np.nan


def compute_num_comm_kernels(df: pd.DataFrame) -> int:
    comm_df = df[df["kind"] == "comm"].copy()
    return int(len(comm_df))


def compute_num_comm_kernels_w_size_and_time(df: pd.DataFrame, n_chips: int) -> int:
    comm_df = df[df["kind"] == "comm"].copy()
    bw_df = comm_df.dropna(subset=["message_bytes", "T_s"]).copy()
    bw_df = bw_df[(bw_df["message_bytes"] > 0) & (bw_df["T_s"] > 0)]
    return int(len(bw_df))


def compute_avg_bandwidth(trace_json_path: str, n_chips: int) -> float :    
    df = prepare_dataframe(trace_json_path)
    comm_df = df[df["kind"] == "comm"].copy()
    _, avg_bandwidth_GBps = compute_comm_bandwidth_table(comm_df, n_chips)
    return avg_bandwidth_GBps


def compute_metric(trace_json_path: str, n_chips: int, metric_type: str = "avg_bandwidth") -> float | int:
    df = prepare_dataframe(trace_json_path)
    
    if metric_type == "avg_bandwidth":
        return compute_avg_bandwidth(trace_json_path, n_chips)
    elif metric_type == "num_comm_kernels":
        return compute_num_comm_kernels(df)
    elif metric_type == "num_comm_kernels_w_size_and_time":
        return compute_num_comm_kernels_w_size_and_time(df, n_chips)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python bandwidth.py <trace_json_path> <n_chips> [avg_bandwidth|num_comm_kernels|num_comm_kernels_w_size_and_time]", file=sys.stderr)
        sys.exit(1)
    trace_path = sys.argv[1]
    n_chips = int(sys.argv[2])
    metric_type = sys.argv[3] if len(sys.argv) > 3 else "avg_bandwidth"
    result = compute_metric(trace_path, n_chips, metric_type)
    print(result)

