"""
Metric: memory_bound_fraction
Description: Percentage of time spent in kernels classified as likely memory-bound based on
             low SM coverage (<50%). These kernels are limited by memory bandwidth rather
             than computational throughput.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable
"""

import sqlite3
import pandas as pd
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
        
        # Load device info for SM count
        device_info = pd.read_sql_query("""
            SELECT numMultiprocessors
            FROM TARGET_INFO_CUDA_DEVICE
        """, conn)
        
        if len(device_info) == 0 or pd.isna(device_info['numMultiprocessors'].iloc[0]):
            conn.close()
            return -1
        
        num_sms = device_info['numMultiprocessors'].iloc[0]
        
        if num_sms is None or num_sms == 0:
            conn.close()
            return -1
        
        # Load kernel data
        kernels = pd.read_sql_query("""
            SELECT 
                (end - start) as duration,
                gridX, gridY, gridZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """, conn)
        
        conn.close()
        
        if len(kernels) == 0:
            return -1
        
        # Calculate blocks per grid and SM coverage
        kernels['blocks_per_grid'] = kernels['gridX'] * kernels['gridY'] * kernels['gridZ']
        kernels['sm_coverage'] = (kernels['blocks_per_grid'] / num_sms).clip(upper=1.0) * 100
        
        # Classify as memory-bound: low SM coverage (<50%)
        memory_bound = kernels['sm_coverage'] < 50
        
        total_time = kernels['duration'].sum()
        memory_bound_time = kernels[memory_bound]['duration'].sum()
        
        if total_time == 0:
            return -1
        
        return float((memory_bound_time / total_time) * 100)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_bound_fraction_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    
    result = calculate_metric(sys.argv[1])
    print(result)
