"""
Metric: communication_fraction
Description: Fraction of GPU time spent in communication kernels (e.g., NCCL, allreduce, 
             allgather). High values indicate communication bottlenecks in distributed training.
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
        
        # Load string mappings
        strings = pd.read_sql_query("SELECT id, value FROM StringIds", conn)
        string_map = dict(zip(strings['id'], strings['value']))
        
        # Load kernel data
        kernels = pd.read_sql_query("""
            SELECT 
                (end - start) as duration,
                demangledName,
                shortName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """, conn)
        
        if len(kernels) == 0:
            conn.close()
            return -1
        
        # Map string IDs to names
        kernels['demangledName_str'] = kernels['demangledName'].map(string_map)
        kernels['shortName_str'] = kernels['shortName'].map(string_map)
        kernels['kernel_name'] = kernels['shortName_str'].fillna(
            kernels['demangledName_str']
        ).fillna('Unknown')
        
        # Identify communication kernels
        comm_patterns = [
            'nccl', 'allreduce', 'allgather', 'reducescatter',
            'broadcast', 'reduce', 'send', 'recv', 'p2p',
            'cross_device', 'communicate', 'all_reduce', 'all_gather'
        ]
        pattern = '|'.join(comm_patterns)
        is_comm = kernels['kernel_name'].str.lower().str.contains(
            pattern, na=False, regex=True
        )
        
        # Calculate fraction
        total_time = kernels['duration'].sum()
        comm_time = kernels[is_comm]['duration'].sum()
        
        conn.close()
        
        if total_time == 0:
            return -1
        
        return float((comm_time / total_time) * 100)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python communication_fraction_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    
    result = calculate_metric(sys.argv[1])
    print(result)
