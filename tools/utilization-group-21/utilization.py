"""
Compute utilization metrics
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

# Import from other metric folders
wall_time_path = os.path.join(tools_dir, "wall_time-group-21", "wall_time.py")
wall_time_spec = importlib.util.spec_from_file_location("wall_time", wall_time_path)
wall_time_module = importlib.util.module_from_spec(wall_time_spec)
wall_time_spec.loader.exec_module(wall_time_module)
compute_wall_time_s = wall_time_module.compute_wall_time_s

compute_time_path = os.path.join(tools_dir, "compute_time-group-21", "compute_time.py")
compute_time_spec = importlib.util.spec_from_file_location("compute_time", compute_time_path)
compute_time_module = importlib.util.module_from_spec(compute_time_spec)
compute_time_spec.loader.exec_module(compute_time_module)
compute_total_compute_time_s = compute_time_module.compute_total_compute_time_s

comm_time_path = os.path.join(tools_dir, "comm_time-group-21", "comm_time.py")
comm_time_spec = importlib.util.spec_from_file_location("comm_time", comm_time_path)
comm_time_module = importlib.util.module_from_spec(comm_time_spec)
comm_time_spec.loader.exec_module(comm_time_module)
compute_total_comm_time_s = comm_time_module.compute_total_comm_time_s


def compute_utilization_proxies(df: pd.DataFrame) -> tuple[float, float]:
    wall_time_s = compute_wall_time_s(df)
    comm_time_s = compute_total_comm_time_s(df)
    comp_time_s = compute_total_compute_time_s(df)
    
    compute_util = comp_time_s / wall_time_s if wall_time_s and wall_time_s > 0 else np.nan
    comm_fraction = comm_time_s / wall_time_s if wall_time_s and wall_time_s > 0 else np.nan
    return float(compute_util), float(comm_fraction)


def compute_compute_utilization(trace_json_path: str) -> float:
    df = prepare_dataframe(trace_json_path)
    compute_util, _ = compute_utilization_proxies(df)
    return compute_util


def compute_communication_fraction(trace_json_path: str) -> float:
    df = prepare_dataframe(trace_json_path)
    _, comm_fraction = compute_utilization_proxies(df)
    return comm_fraction


def compute_metric(trace_json_path: str, metric_type: str = "compute_util") -> float:
    if metric_type == "compute_util":
        return compute_compute_utilization(trace_json_path)
    elif metric_type == "comm_fraction":
        return compute_communication_fraction(trace_json_path)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utilization.py <trace_json_path> [compute_util|comm_fraction]", file=sys.stderr)
        sys.exit(1)
    
    trace_path = sys.argv[1]
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "compute_util"
    result = compute_metric(trace_path, metric_type)
    print(result)

