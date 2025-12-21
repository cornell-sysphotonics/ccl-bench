#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Usage:
    python traffic_interval_analyzer.py <nsys_directory_or_file>
    python traffic_interval_analyzer.py ../../nsys_1node
"""

import os
import re
import sqlite3
import subprocess
import numpy as np
from collections import defaultdict


ITERATION_PATTERNS_WITH_NUM = [
    r'train\s+step\s+(\d+)',           # "Train Step 0", "train step 1"
    r'training\s+step\s+(\d+)',        # "training step 0"
    r'iteration\s+(\d+)',              # "iteration 0", "Iteration 1"
    r'^step\s+(\d+)$',                 # "step 0"
    r'^Train Step (\d+)$',             # "Train Step 0"
]

ITERATION_PATTERNS_NO_NUM = [
    r'^DeepSpeedZeroOptimizer_Stage3\.step$',  # DeepSpeed ZeRO-3 optimizer step
    r'^DeepSpeedZeroOptimizer_Stage2\.step$',  # DeepSpeed ZeRO-2 optimizer step
    r'^optimizer\.step$',                       # Generic optimizer step
    r'^DeepSpeedEngine\.forward$',              # DeepSpeed forward (fallback for ZeRO-2)
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
        list: [{'iter_num': int, 'start': ns, 'end': ns, 'name': str}, ...]
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
        return iterations
    
    if unnumbered_markers:
        from collections import Counter
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
            
            iterations.append({
                'iter_num': i,
                'start': start,
                'end': end,
                'name': f"{main_markers[i]['name']} (auto #{i})"
            })
        
        return iterations
    
    return []


def get_collective_type(kernel_name):
    """从 kernel 名称中提取集合操作类型"""
    name_lower = kernel_name.lower()
    
    if 'allgather' in name_lower or 'all_gather' in name_lower:
        return 'AllGather'
    elif 'reducescatter' in name_lower or 'reduce_scatter' in name_lower:
        return 'ReduceScatter'
    elif 'allreduce' in name_lower or 'all_reduce' in name_lower:
        return 'AllReduce'
    elif 'broadcast' in name_lower:
        return 'Broadcast'
    elif 'alltoall' in name_lower or 'all_to_all' in name_lower:
        return 'AllToAll'
    elif 'send' in name_lower or 'recv' in name_lower:
        return 'SendRecv'
    else:
        return 'Other'


def calc_stats(values_ms):
    if not values_ms:
        return None
    
    arr = np.array(values_ms)
    return {
        'count': len(arr),
        'mean_ms': float(np.mean(arr)),
        'std_ms': float(np.std(arr)),
        'min_ms': float(np.min(arr)),
        'max_ms': float(np.max(arr)),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99))
    }


def calc_interval_stats(events):

    if len(events) < 2:
        return None
    
    intervals_ns = []
    for i in range(len(events) - 1):
        interval = events[i + 1]['start'] - events[i]['end']
        intervals_ns.append(interval)
    
    intervals_ms = [i / 1e6 for i in intervals_ns]
    
    positive = [i for i in intervals_ms if i > 0]
    negative = [i for i in intervals_ms if i < 0]
    
    stats = calc_stats(intervals_ms)
    if stats:
        stats['positive_count'] = len(positive)
        stats['negative_count'] = len(negative)
        stats['positive_mean_ms'] = float(np.mean(positive)) if positive else 0.0
        stats['negative_mean_ms'] = float(np.mean(negative)) if negative else 0.0
    
    return stats


def analyze_traffic_intervals(nsys_rep_file):
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing traffic intervals (per-iteration) in {nsys_rep_file}...")
    
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    temp_sqlite = False
    
    if not os.path.exists(sqlite_file):
        sqlite_file = nsys_rep_file.replace('.nsys-rep', '_temp.sqlite')
        temp_sqlite = True
        
        print(f"  Exporting to SQLite...")
        try:
            export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}",
                         "--force-overwrite=true", nsys_rep_file]
            subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            
            if not os.path.exists(sqlite_file):
                print(f"  Failed to export SQLite")
                return None
        except subprocess.TimeoutExpired:
            print(f"  Timeout exporting SQLite")
            return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        iterations = get_iteration_boundaries(cursor)
        print(f"  Found {len(iterations)} iteration markers")
        
        if len(iterations) < 2:
            print("  Warning: Not enough iteration markers found, analyzing whole trace")
            return analyze_whole_trace(cursor, conn, sqlite_file, temp_sqlite)
        
        query = """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE '%nccl%'
        ORDER BY k.start
        """
        
        cursor.execute(query)
        kernels = cursor.fetchall()
        conn.close()
        
        if temp_sqlite and os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        
        if not kernels:
            print("  No NCCL kernels found")
            return None
        
        print(f"  Found {len(kernels)} NCCL events")
        
        iter_events = defaultdict(list)
        
        for start, end, name in kernels:
            coll_type = get_collective_type(name)
            event = {
                'start': start,
                'end': end,
                'duration_ns': end - start,
                'type': coll_type
            }
            
            for it in iterations:
                if it['start'] <= start <= it['end']:
                    iter_events[it['iter_num']].append(event)
                    break
        
        per_iteration_stats = {}
        
        all_durations = defaultdict(list)
        all_intervals = defaultdict(list)
        
        for iter_num in sorted(iter_events.keys()):
            events = iter_events[iter_num]
            
            by_type = defaultdict(list)
            for e in events:
                by_type[e['type']].append(e)
            
            iter_stats = {
                'total_events': len(events),
                'per_operation': {}
            }
            
            for coll_type, type_events in by_type.items():
                type_events.sort(key=lambda x: x['start'])
                
                durations_ms = [e['duration_ns'] / 1e6 for e in type_events]
                duration_stats = calc_stats(durations_ms)
                
                if duration_stats:
                    duration_stats['total_time_ms'] = sum(durations_ms)
                
                interval_stats = calc_interval_stats(type_events)
                
                iter_stats['per_operation'][coll_type] = {
                    'duration': duration_stats,
                    'interval': interval_stats
                }
                
                all_durations[coll_type].extend(durations_ms)
                if interval_stats and interval_stats['count'] > 0:
                    all_intervals[coll_type].append(interval_stats['mean_ms'])
            
            per_iteration_stats[f"iter_{iter_num}"] = iter_stats
        
        summary = {
            'num_iterations': len(per_iteration_stats),
            'total_nccl_events': len(kernels),
            'events_in_iterations': sum(
                s['total_events'] for s in per_iteration_stats.values()
            ),
            'per_operation': {}
        }
        
        for coll_type in all_durations.keys():
            durations = all_durations[coll_type]
            intervals = all_intervals[coll_type]
            
            summary['per_operation'][coll_type] = {
                'total_calls': len(durations),
                'calls_per_iteration': len(durations) / len(per_iteration_stats) if per_iteration_stats else 0,
                'duration': calc_stats(durations),
                'interval_mean_across_iters': {
                    'mean_ms': float(np.mean(intervals)) if intervals else None,
                    'std_ms': float(np.std(intervals)) if intervals else None,
                    'min_ms': float(np.min(intervals)) if intervals else None,
                    'max_ms': float(np.max(intervals)) if intervals else None,
                } if intervals else None
            }
            
            if summary['per_operation'][coll_type]['duration']:
                summary['per_operation'][coll_type]['duration']['total_time_ms'] = sum(durations)
        
        results = {
            'analysis_mode': 'per_iteration',
            'num_iterations': len(per_iteration_stats),
            'iteration_markers': [
                {'iter': it['iter_num'], 'name': it['name']} 
                for it in iterations[:5]
            ],
            'summary': summary,
            'per_iteration': per_iteration_stats
        }
        
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        if temp_sqlite and os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return None


def analyze_whole_trace(cursor, conn, sqlite_file, temp_sqlite):
    print("  Falling back to whole-trace analysis...")
    
    query = """
    SELECT k.start, k.end, s.value as name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    WHERE s.value LIKE '%nccl%'
    ORDER BY k.start
    """
    
    cursor.execute(query)
    kernels = cursor.fetchall()
    conn.close()
    
    if temp_sqlite and os.path.exists(sqlite_file):
        os.remove(sqlite_file)
    
    if not kernels:
        return None
    
    by_type = defaultdict(list)
    
    for start, end, name in kernels:
        coll_type = get_collective_type(name)
        by_type[coll_type].append({
            'start': start,
            'end': end,
            'duration_ns': end - start
        })
    
    results = {
        'analysis_mode': 'whole_trace',
        'total_nccl_events': len(kernels),
        'per_operation': {}
    }
    
    for coll_type, events in by_type.items():
        events.sort(key=lambda x: x['start'])
        
        durations_ms = [e['duration_ns'] / 1e6 for e in events]
        duration_stats = calc_stats(durations_ms)
        if duration_stats:
            duration_stats['total_time_ms'] = sum(durations_ms)
        
        interval_stats = calc_interval_stats(events)
        
        results['per_operation'][coll_type] = {
            'duration': duration_stats,
            'interval': interval_stats
        }
    
    return results


def analyze_directory(nsys_path):
    nsys_files = []
    
    if os.path.isfile(nsys_path):
        if nsys_path.endswith('.nsys-rep'):
            nsys_files.append(nsys_path)
    elif os.path.isdir(nsys_path):
        for f in os.listdir(nsys_path):
            if f.endswith('.nsys-rep'):
                nsys_files.append(os.path.join(nsys_path, f))
    
    if not nsys_files:
        print(f"No .nsys-rep files found")
        return None
    
    print(f"Found {len(nsys_files)} nsys-rep file(s)\n")
    
    all_results = []
    for nsys_file in sorted(nsys_files):
        result = analyze_traffic_intervals(nsys_file)
        if result:
            result['file'] = os.path.basename(nsys_file)
            all_results.append(result)
    
    if len(all_results) == 1:
        return all_results[0]
    
    merged = all_results[0].copy()
    merged['num_files'] = len(all_results)
    return merged


def print_results(results):
    print("\n" + "=" * 80)
    print("Traffic Interval Analysis - Per-Iteration")
    print("=" * 80)
    
    mode = results.get('analysis_mode', 'unknown')
    print(f"Analysis mode: {mode}")
    
    if mode == 'per_iteration':
        print(f"Number of iterations: {results['num_iterations']}")
        
        if results.get('iteration_markers'):
            print(f"Iteration markers (sample): {[m['name'] for m in results['iteration_markers'][:3]]}")
        
        summary = results.get('summary', {})
        print(f"Total NCCL events: {summary.get('total_nccl_events', 'N/A')}")
        print(f"Events in iterations: {summary.get('events_in_iterations', 'N/A')}")
        
        print(f"\n{'─' * 80}")
        print("📊 Summary Statistics (across all iterations)")
        print(f"{'─' * 80}")
        
        for coll_type, data in sorted(summary.get('per_operation', {}).items(),
                                       key=lambda x: x[1].get('total_calls', 0),
                                       reverse=True):
            print(f"\n  {coll_type}:")
            print(f"    Total calls: {data.get('total_calls', 'N/A')}")
            print(f"    Calls per iteration: {data.get('calls_per_iteration', 0):.1f}")
            
            dur = data.get('duration')
            if dur:
                print(f"    Duration - Mean: {dur['mean_ms']:.4f} ms, P50: {dur['p50_ms']:.4f} ms, P99: {dur['p99_ms']:.4f} ms")
            
            intv = data.get('interval_mean_across_iters')
            if intv and intv.get('mean_ms') is not None:
                print(f"    Interval (mean across iters) - Mean: {intv['mean_ms']:.4f} ms, Std: {intv['std_ms']:.4f} ms")
        
        per_iter = results.get('per_iteration', {})
        if per_iter:
            sample_iter = list(per_iter.keys())[0]
            sample_data = per_iter[sample_iter]
            
            print(f"\n{'─' * 80}")
            print(f"📊 Sample Iteration: {sample_iter}")
            print(f"{'─' * 80}")
            print(f"  Total events: {sample_data.get('total_events', 'N/A')}")
            
            for coll_type, op_data in sample_data.get('per_operation', {}).items():
                dur = op_data.get('duration')
                intv = op_data.get('interval')
                
                if dur:
                    print(f"\n  {coll_type}:")
                    print(f"    Calls: {dur['count']}")
                    print(f"    Duration - Mean: {dur['mean_ms']:.4f} ms, Total: {dur.get('total_time_ms', 0):.2f} ms")
                    
                    if intv:
                        print(f"    Interval - Mean: {intv['mean_ms']:.4f} ms, Pos: {intv['positive_count']}, Neg: {intv['negative_count']}")
    
    else:
        print(f"Total NCCL events: {results.get('total_nccl_events', 'N/A')}")
        
        for coll_type, data in results.get('per_operation', {}).items():
            dur = data.get('duration')
            intv = data.get('interval')
            
            print(f"\n  {coll_type}:")
            if dur:
                print(f"    Calls: {dur['count']}, Total: {dur.get('total_time_ms', 0):.2f} ms")
                print(f"    Duration - Mean: {dur['mean_ms']:.4f} ms, P99: {dur['p99_ms']:.4f} ms")
            if intv:
                print(f"    Interval - Mean: {intv['mean_ms']:.4f} ms")
    
    print("\n" + "=" * 80)


def metric_cal(directory):
    results = analyze_directory(directory)
    if results:
        print_results(results)
    return results


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        results = metric_cal(path)
        
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
    else:
        print("Usage: python traffic_interval_analyzer.py <nsys_directory_or_file> [output.json]")
