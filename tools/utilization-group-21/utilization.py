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
    """
    Compute utilization metrics accounting for parallel execution across TPU cores.
    
    For multi-device TPU traces, we calculate per-device fractions and average them
    to avoid double-counting parallel operations.
    """
    # Get overall wall time
    wall_time_s = compute_wall_time_s(df)
    
    # Check if we have multiple devices (TPU cores)
    if 'pid' in df.columns and df['pid'].nunique() > 1:
        # Per-device calculation for parallel execution
        compute_fractions = []
        comm_fractions = []
        
        for pid in df['pid'].unique():
            device_df = df[df['pid'] == pid].copy()
            
            # Device wall time
            dev_wall = device_df['end_s'].max() - device_df['start_s'].min()
            if dev_wall <= 0:
                continue
            
            # Device comm time
            comm_df = device_df[device_df['kind'] == 'comm']
            dev_comm_time = np.nansum(comm_df['T_s']) if comm_df['T_s'].notna().any() else comm_df['dur_s'].sum()
            
            # Device compute time
            comp_df = device_df[device_df['kind'] == 'compute']
            dev_comp_time = np.nansum(comp_df['T_s']) if comp_df['T_s'].notna().any() else comp_df['dur_s'].sum()
            
            # Skip PIDs with no compute or comm work (e.g., host/coordinator processes)
            if dev_comp_time == 0 and dev_comm_time == 0:
                continue
            
            compute_fractions.append(dev_comp_time / dev_wall)
            comm_fractions.append(dev_comm_time / dev_wall)
        
        compute_util = float(np.mean(compute_fractions)) if compute_fractions else np.nan
        comm_fraction = float(np.mean(comm_fractions)) if comm_fractions else np.nan
        
    else:
        # Single device or no pid info - use original calculation
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

