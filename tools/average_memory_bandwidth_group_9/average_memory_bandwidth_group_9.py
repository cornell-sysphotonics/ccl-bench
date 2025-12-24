"""
Metric: average_memory_bandwidth
Description: Average memory bandwidth achieved during memory copy operations. Lower values
             compared to hardware peak indicate inefficient memory access patterns or small
             transfer sizes.
Unit: GB/s (Gigabytes per second)
Returns: Float (GB/s), or -1 if data unavailable
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
        
        # Check if memcpy table exists
        tables = pd.read_sql_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'
        """, conn)
        
        if len(tables) == 0:
            conn.close()
            return -1
        
        # Load memory copy data
        memcpy = pd.read_sql_query("""
            SELECT 
                (end - start) as duration,
                bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """, conn)
        
        conn.close()
        
        if len(memcpy) == 0:
            return -1
        
        # Calculate bandwidth in GB/s for each transfer
        # bytes / nanoseconds * 1e9 / 1e9 = GB/s
        memcpy['bandwidth_GBs'] = (memcpy['bytes'] / memcpy['duration']) * 1e9 / 1e9
        
        # Filter out invalid values
        memcpy = memcpy[memcpy['bandwidth_GBs'] > 0]
        
        if len(memcpy) == 0:
            return -1
        
        return float(memcpy['bandwidth_GBs'].mean())
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python average_memory_bandwidth_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    
    result = calculate_metric(sys.argv[1])
    print(result)
