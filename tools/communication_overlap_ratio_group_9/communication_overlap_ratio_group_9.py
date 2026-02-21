"""
Metric: communication_overlap_ratio
Description: Ratio of overlapping communication kernels to total communication kernels.
             Higher values indicate better overlap between communication operations, which
             can improve overall throughput in distributed training.
Unit: Ratio (0-1)
Returns: Float between 0-1, or -1 if data unavailable
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
                start, end,
                demangledName,
                shortName,
                deviceId,
                streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """, conn)
        
        conn.close()
        
        if len(kernels) == 0:
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
        
        comm_kernels = kernels[is_comm].copy()
        
        if len(comm_kernels) <= 1:
            return -1
        
        # Sort by start time
        comm_kernels = comm_kernels.sort_values('start')
        
        # Count overlaps (when a kernel starts before the previous one ends)
        overlaps = 0
        total_pairs = 0
        
        for device_id in comm_kernels['deviceId'].unique():
            device_kernels = comm_kernels[comm_kernels['deviceId'] == device_id].sort_values('start')
            
            if len(device_kernels) <= 1:
                continue
            
            for i in range(len(device_kernels) - 1):
                current_end = device_kernels.iloc[i]['end']
                next_start = device_kernels.iloc[i + 1]['start']
                
                if next_start < current_end:
                    overlaps += 1
                
                total_pairs += 1
        
        if total_pairs == 0:
            return -1
        
        return float(overlaps / total_pairs)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python communication_overlap_ratio_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    
    result = calculate_metric(sys.argv[1])
    print(result)
