#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze iteration time from nsys traces

METRIC 0 Implementation:
  - iteration_time_mean: Mean wall-clock duration of one training iteration
  - iteration_time_p99: 99th percentile of iteration time
"""

import subprocess
import sqlite3
import re
import os
import numpy as np
from collections import Counter

ITERATION_PATTERNS_WITH_NUM = [
    r'train\s+step\s+(\d+)',           # "Train Step 0", "train step 1"
    r'training\s+step\s+(\d+)',        # "training step 0"
    r'iteration\s+(\d+)',              # "iteration 0", "Iteration 1"
    r'^step\s+(\d+)$',                 # "step 0"
    r'Train Step (\d+)',               # "Train Step 0"
]

ITERATION_PATTERNS_NO_NUM = [
    r'^DeepSpeedZeroOptimizer_Stage3\.step$',  # DeepSpeed ZeRO-3 optimizer step
    r'^DeepSpeedZeroOptimizer_Stage2\.step$',  # DeepSpeed ZeRO-2 optimizer step
    r'^optimizer\.step$',                       # Generic optimizer step
    r'^DeepSpeedEngine\.forward$',              # DeepSpeed forward (fallback)
]

EXCLUDE_PATTERNS = [
    r'_post_step',
    r'backward_step',
    r'reduce_step',
    r'_step$',
]


def is_valid_iteration_marker(name):
    """    
    Returns:
        tuple: (is_valid, iter_num_or_none, needs_numbering)
    """
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, name):
            return False, None, False
    
    name_lower = name.lower()
    for pattern in ITERATION_PATTERNS_WITH_NUM:
        match = re.search(pattern, name_lower, re.IGNORECASE)
        if match:
            iter_num = int(match.group(1))
            return True, iter_num, False
    
    for pattern in ITERATION_PATTERNS_NO_NUM:
        if re.match(pattern, name):
            return True, None, True
    
    return False, None, False


def get_iteration_boundaries(cursor):
    """
    
    Returns:
        list: [{'iter_num': int, 'start': ns, 'end': ns, 'name': str, 'duration_ms': float}, ...]
    """
    try:
        query = """
        SELECT start, end, text, globalTid
        FROM NVTX_EVENTS
        WHERE text IS NOT NULL 
          AND start IS NOT NULL 
          AND end IS NOT NULL
        ORDER BY start
        """
        cursor.execute(query)
        ranges = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"  NVTX query failed: {e}")
        return []
    
    numbered_markers = []
    unnumbered_markers = []
    
    for start, end, name, tid in ranges:
        if name is None:
            continue
        name_str = str(name)
        is_valid, iter_num, needs_numbering = is_valid_iteration_marker(name_str)
        
        if is_valid:
            if needs_numbering:
                unnumbered_markers.append({
                    'start': start,
                    'end': end,
                    'name': name_str,
                    'tid': tid
                })
            else:
                numbered_markers.append({
                    'iter_num': iter_num,
                    'start': start,
                    'end': end,
                    'name': name_str,
                    'tid': tid
                })
    
    if numbered_markers:
        iter_map = {}
        for m in numbered_markers:
            iter_num = m['iter_num']
            if iter_num not in iter_map:
                iter_map[iter_num] = {
                    'iter_num': iter_num,
                    'start': m['start'],
                    'end': m['end'],
                    'name': m['name']
                }
            else:
                iter_map[iter_num]['start'] = min(iter_map[iter_num]['start'], m['start'])
                iter_map[iter_num]['end'] = max(iter_map[iter_num]['end'], m['end'])
        
        iterations = list(iter_map.values())
        iterations.sort(key=lambda x: x['iter_num'])
        
        for it in iterations:
            it['duration_ms'] = (it['end'] - it['start']) / 1e6
        
        return iterations
    
    if unnumbered_markers:
        tid_counts = Counter(m['tid'] for m in unnumbered_markers)
        main_tid = tid_counts.most_common(1)[0][0]
        
        main_markers = [m for m in unnumbered_markers if m['tid'] == main_tid]
        main_markers.sort(key=lambda x: x['start'])
        
        iterations = []
        for i in range(len(main_markers)):
            start = main_markers[i]['start']
            if i < len(main_markers) - 1:
                end = main_markers[i + 1]['start'] - 1
            else:
                end = main_markers[i]['end']
            
            duration_ms = (end - start) / 1e6
            iterations.append({
                'iter_num': i,
                'start': start,
                'end': end,
                'name': f"{main_markers[i]['name']} (auto #{i})",
                'duration_ms': duration_ms
            })
        
        return iterations
    
    return []


def export_to_sqlite(nsys_rep_file):
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    
    if os.path.exists(sqlite_file):
        return sqlite_file
    
    print(f"  Exporting to SQLite...")
    nsys_cmd = os.environ.get('NSYS', 'nsys')
    cmd = [nsys_cmd, 'export', '--type=sqlite', f'--output={sqlite_file}', 
           '--force-overwrite=true', nsys_rep_file]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if os.path.exists(sqlite_file):
            return sqlite_file
    except Exception as e:
        print(f"  Export failed: {e}")
    
    return None


def analyze_iteration_time(nsys_rep_file):
    """
    
    Returns:
        dict: Iteration time statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing iteration times in {nsys_rep_file}...")
    
    if nsys_rep_file.endswith('.sqlite'):
        sqlite_file = nsys_rep_file
    else:
        sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
        if not os.path.exists(sqlite_file):
            sqlite_file = export_to_sqlite(nsys_rep_file)
            if not sqlite_file:
                print("  Failed to get SQLite file")
                return None
    
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    iterations = get_iteration_boundaries(cursor)
    conn.close()
    
    if not iterations:
        print("  ✗ No valid iteration markers found")
        print("  Hint: Make sure training script uses NVTX markers")
        return None
    
    marker_type = iterations[0]['name'].split('(')[0].strip() if '(auto' in iterations[0]['name'] else iterations[0]['name']
    print(f"  ✓ Found {len(iterations)} iterations using marker: {marker_type}")
    
    iteration_times_ms = [it['duration_ms'] for it in iterations]
    
    stats = {
        "iteration_time_mean": np.mean(iteration_times_ms),
        "iteration_time_p99": np.percentile(iteration_times_ms, 99),
        "avg_iteration_time_ms": np.mean(iteration_times_ms),
        "p50_iteration_time_ms": np.percentile(iteration_times_ms, 50),
        "p99_iteration_time_ms": np.percentile(iteration_times_ms, 99),
        "min_iteration_time_ms": np.min(iteration_times_ms),
        "max_iteration_time_ms": np.max(iteration_times_ms),
        "std_iteration_time_ms": np.std(iteration_times_ms),
        "num_iterations": len(iteration_times_ms),
        "iteration_times_ms": iteration_times_ms,
        "marker_type": marker_type
    }
    
    print(f"\n  === METRIC 0 Summary ===")
    print(f"  Number of iterations: {stats['num_iterations']}")
    print(f"  iteration_time_mean: {stats['iteration_time_mean']:.2f} ms")
    print(f"  iteration_time_p99: {stats['iteration_time_p99']:.2f} ms")
    
    return stats


def metric_cal(directory):
    """
    Calculate iteration time metrics
    
    Args:
        directory: Trace directory, nsys directory, or .nsys-rep/.sqlite file path
    
    Returns:
        float: Average iteration time in milliseconds
    """
    nsys_file = None
    
    if os.path.isfile(directory):
        if directory.endswith('.nsys-rep') or directory.endswith('.sqlite'):
            nsys_file = directory
    
    elif os.path.isdir(directory):
        for f in os.listdir(directory):
            if f.endswith('.sqlite'):
                nsys_file = os.path.join(directory, f)
                break
        if not nsys_file:
            for f in os.listdir(directory):
                if f.endswith('.nsys-rep'):
                    nsys_file = os.path.join(directory, f)
                    break
    
    if not nsys_file:
        print("No nsys-rep or sqlite files found")
        return 0.0
    
    stats = analyze_iteration_time(nsys_file)
    
    if stats:
        print("\n" + "="*60)
        print("Iteration Time Analysis Results")
        print("="*60)
        print(f"Marker type: {stats.get('marker_type', 'N/A')}")
        print(f"Number of iterations: {stats['num_iterations']}")
        print(f"Average iteration time: {stats['avg_iteration_time_ms']:.2f} ms")
        print(f"Median (P50) iteration time: {stats['p50_iteration_time_ms']:.2f} ms")
        print(f"P99 iteration time: {stats['p99_iteration_time_ms']:.2f} ms")
        print(f"Min iteration time: {stats['min_iteration_time_ms']:.2f} ms")
        print(f"Max iteration time: {stats['max_iteration_time_ms']:.2f} ms")
        print(f"Std deviation: {stats['std_iteration_time_ms']:.2f} ms")
        print("="*60)
        
        return stats['avg_iteration_time_ms']
    
    return 0.0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        print(f"\nAverage iteration time: {result:.2f} ms")
    else:
        print("Usage: python iteration_time_analyzer.py <nsys_directory_or_file>")
