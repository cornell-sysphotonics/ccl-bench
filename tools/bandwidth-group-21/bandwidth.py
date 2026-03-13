"""
Compute bandwidth metrics
"""
import sys
import os
import importlib.util
import numpy as np
import pandas as pd
import yaml

# Import utils-group-21
tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(tools_dir, "utils-group-21.py")
spec = importlib.util.spec_from_file_location("utils_group_21", utils_path)
utils_group_21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_group_21)
prepare_dataframe = utils_group_21.prepare_dataframe

def get_n_chips_from_card(trace_json_path):
    base = trace_json_path
    for ext in (".trace.json", ".json.gz", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    yaml_path = base + ".yaml"

    if not os.path.exists(yaml_path):
        folder = os.path.dirname(trace_json_path)
        yaml_files = [f for f in os.listdir(folder) if f.endswith('.yaml')]
        if not yaml_files:
            return None
        yaml_path = os.path.join(folder, yaml_files[0])

    with open(yaml_path, "r") as f:
        card = yaml.safe_load(f)
    try:
        return card["workload"]["hardware"]["xpu_spec"]["total_count"]
    except Exception as e:
        return None

def compute_comm_bandwidth_table(comm_df: pd.DataFrame, n_chips: int) -> tuple[pd.DataFrame, float]:
    bw_df = comm_df.dropna(subset=["message_bytes", "T_s"]).copy()
    bw_df = bw_df[(bw_df["message_bytes"] > 0) & (bw_df["T_s"] > 0)]
    bw_df["bandwidth_Bps"] = (n_chips - 1) * bw_df["message_bytes"] / bw_df["T_s"]
    bw_df["bandwidth_GBps"] = bw_df["bandwidth_Bps"] / 1e9
    
    # Calculate average bandwidth using total bytes / total time to avoid outliers from tiny T_s values
    total_bytes = bw_df["message_bytes"].sum()
    total_time = bw_df["T_s"].sum()
    avg_bandwidth_GBps = (n_chips - 1) * total_bytes / total_time / 1e9 if total_time > 0 else np.nan
    
    return bw_df, float(avg_bandwidth_GBps) if np.isfinite(avg_bandwidth_GBps) else np.nan


def compute_num_comm_kernels(trace_json_path: str) -> int:
    n_chips_card = get_n_chips_from_card(trace_json_path)
    if n_chips_card is None:
        raise ValueError("n_chips not found in workload card YAML")
    df = prepare_dataframe(trace_json_path)
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


def compute_metric(trace_json_path: str, metric_type: str = "avg_bandwidth") -> float | int:
    n_chips_card = get_n_chips_from_card(trace_json_path)
    if n_chips_card is None:
        raise ValueError("n_chips not found in workload card YAML")
    if metric_type == "avg_bandwidth":
        return compute_avg_bandwidth(trace_json_path, n_chips_card)
    elif metric_type == "num_comm_kernels":
        return compute_num_comm_kernels(trace_json_path)
    elif metric_type == "num_comm_kernels_w_size_and_time":
        df = prepare_dataframe(trace_json_path)
        return compute_num_comm_kernels_w_size_and_time(df, n_chips_card)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bandwidth.py <trace_json_path> [avg_bandwidth|num_comm_kernels|num_comm_kernels_w_size_and_time]", file=sys.stderr)
        sys.exit(1)
    trace_path = sys.argv[1]
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "avg_bandwidth"
    result = compute_metric(trace_path, None, metric_type)
    print(result)

