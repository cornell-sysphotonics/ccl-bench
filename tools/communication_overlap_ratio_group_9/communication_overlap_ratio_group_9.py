"""
Metric: communication_overlap_ratio
Description: Fraction of communication time that is overlapped (hidden) by concurrent
             computation. Higher values indicate better pipelining of communication and
             compute, reducing the effective cost of communication.
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
            'broadcast', 'send', 'recv', 'p2p',
            'cross_device', 'communicate', 'all_reduce', 'all_gather'
        ]
        pattern = '|'.join(comm_patterns)
        is_comm = kernels['kernel_name'].str.lower().str.contains(
            pattern, na=False, regex=True
        )

        comm_kernels = kernels[is_comm]
        compute_kernels = kernels[~is_comm]

        if len(comm_kernels) == 0:
            return -1

        # Helper: merge overlapping intervals and return list of (start, end)
        def _merge_intervals(starts, ends):
            intervals = sorted(zip(starts, ends))
            merged = [intervals[0]]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            return merged

        # Helper: compute total intersection time between two merged interval lists
        def _intersection_time(a_intervals, b_intervals):
            total = 0
            i = j = 0
            while i < len(a_intervals) and j < len(b_intervals):
                start = max(a_intervals[i][0], b_intervals[j][0])
                end = min(a_intervals[i][1], b_intervals[j][1])
                if start < end:
                    total += end - start
                if a_intervals[i][1] < b_intervals[j][1]:
                    i += 1
                else:
                    j += 1
            return total

        total_comm_time = 0
        total_overlap_time = 0

        for device_id in comm_kernels['deviceId'].unique():
            dev_comm = comm_kernels[comm_kernels['deviceId'] == device_id]
            dev_compute = compute_kernels[compute_kernels['deviceId'] == device_id]

            comm_merged = _merge_intervals(dev_comm['start'].values, dev_comm['end'].values)
            comm_time = sum(e - s for s, e in comm_merged)
            total_comm_time += comm_time

            if len(dev_compute) == 0:
                continue

            compute_merged = _merge_intervals(dev_compute['start'].values, dev_compute['end'].values)
            total_overlap_time += _intersection_time(comm_merged, compute_merged)

        if total_comm_time == 0:
            return -1

        return float(total_overlap_time / total_comm_time)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python communication_overlap_ratio_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    
    result = calculate_metric(sys.argv[1])
    print(result)
