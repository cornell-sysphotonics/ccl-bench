"""
Metric: load_imbalance_ratio
Description: Ratio of maximum to minimum GPU execution time in multi-GPU setups. Values close
             to 1.0 indicate balanced workload distribution, while higher values (>1.2) suggest
             load imbalance where some GPUs are idle while others are busy.
Unit: Ratio (dimensionless)
Returns: Float >= 1.0, or -1 if data unavailable or single GPU
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os


def find_sqlite_file(path):
    """Find SQLite file in directory or return path if it's already a .sqlite file"""
    # Convert to absolute path to avoid any relative path issues
    path = os.path.abspath(path)
    
    if os.path.isfile(path) and path.endswith('.sqlite'):
        return path
    
    if os.path.isdir(path):
        sqlite_files = [f for f in os.listdir(path) if f.endswith('.sqlite')]
        if len(sqlite_files) == 0:
            return None
        # Prefer non-profiling files
        non_profiling = [f for f in sqlite_files if 'profiling' not in f.lower()]
        if non_profiling:
            return os.path.abspath(os.path.join(path, non_profiling[0]))
        return os.path.abspath(os.path.join(path, sqlite_files[0]))
    
    return None


def calculate_metric(path):
    """
    Calculate metric from SQLite trace file.
    
    Args:
        path: Either a directory containing .sqlite file or direct path to .sqlite file
    
    Returns:
        float: Metric value, or -1 if calculation fails
    """
    # Find the SQLite file
    sqlite_path = find_sqlite_file(path)
    if sqlite_path is None:
        print(f"Error: No .sqlite file found in {path}", file=sys.stderr)
        return -1
    
    try:
        conn = sqlite3.connect(sqlite_path)
        
        # Load kernel data with device IDs
        kernels = pd.read_sql_query("""
            SELECT 
                deviceId,
                (end - start) as duration
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """, conn)
        
        conn.close()
        
        if len(kernels) == 0:
            return -1
        
        # Get unique GPUs
        unique_gpus = kernels['deviceId'].unique()
        
        if len(unique_gpus) <= 1:
            # Single GPU or no data - no imbalance possible
            return -1
        
        # Calculate total execution time per GPU
        gpu_times = []
        for gpu_id in unique_gpus:
            gpu_kernels = kernels[kernels['deviceId'] == gpu_id]
            total_time = gpu_kernels['duration'].sum()
            gpu_times.append(total_time)
        
        gpu_times = np.array(gpu_times)
        
        if gpu_times.min() == 0:
            # Avoid division by zero
            return -1
        
        # Calculate max/min ratio
        imbalance_ratio = gpu_times.max() / gpu_times.min()
        
        return float(imbalance_ratio)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_imbalance_ratio_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    
    result = calculate_metric(sys.argv[1])
    print(result)
