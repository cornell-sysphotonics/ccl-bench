#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 2: Parallelism-Phase Window Time Analyzer

Measures time gaps (windows) between different parallelism phases.

Definition (from spec):
  For adjacent phases P1 (tag=X) and P2 (tag=Y):
    window(P1→P2) = start_time(P2) - end_time(P1)

Interpretation:
  - window > 0: idle time (gap) between phases
  - window < 0: overlap between phases (good for hiding latency)

Output:
  For each transition type (X→Y):
    - mean_window, p50_window, p95_window, count
"""

import subprocess
import os
import sqlite3
from collections import defaultdict
import numpy as np


def export_to_sqlite(nsys_rep_file):
    """Export nsys-rep to SQLite for detailed timeline analysis"""
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '_temp.sqlite')
    
    try:
        export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}", 
                     "--force-overwrite=true", nsys_rep_file]
        result = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        
        if os.path.exists(sqlite_file):
            return sqlite_file
        return None
    except Exception as e:
        print(f"  Error exporting to SQLite: {e}")
        return None


def categorize_nccl_kernel(kernel_name, tp_size=1, pp_size=1, dp_size=1, ep_size=1):
    """
    Categorize NCCL kernel by parallelism type (DP/TP/PP/EP)
    
    Heuristic rules for DeepSpeed ZeRO-3:
      - ReduceScatter: DP (ZeRO gradient partitioning)
      - AllGather: DP (ZeRO parameter gathering) - NOT TP!
      - AllReduce: DP (gradient aggregation) or TP (tensor parallel reduce)
      - Send/Recv: PP (pipeline parallel)
      - AllToAll: EP (expert parallel for MoE)
      - Broadcast: OTHER (initialization, etc.)
    
    Note: This is the same logic as comm_time_breakdown.py for consistency.
    
    Args:
        kernel_name: NCCL kernel name string
        tp_size: Tensor parallel size (for future use with config)
        pp_size: Pipeline parallel size
        dp_size: Data parallel size
        ep_size: Expert parallel size
    
    Returns:
        str: Parallelism type ('DP', 'TP', 'PP', 'EP', 'OTHER')
    """
    name_lower = kernel_name.lower()
    
    # 1. Send/Recv -> PP (Pipeline Parallel)
    if 'send' in name_lower or 'recv' in name_lower:
        return 'PP'
    
    # 2. AllToAll -> EP (Expert Parallel for MoE)
    if 'alltoall' in name_lower:
        return 'EP'
    
    # 3. ReduceScatter -> DP (ZeRO gradient partitioning)
    if 'reducescatter' in name_lower:
        return 'DP'
    
    # 4. AllGather -> DP (ZeRO parameter gathering)
    #    Note: In ZeRO-3, AllGather is used to gather partitioned parameters
    #    This is DP communication, not TP!
    if 'allgather' in name_lower:
        return 'DP'
    
    # 5. AllReduce -> Could be DP or TP
    #    Without config, assume AllReduce is DP (gradient aggregation)
    if 'allreduce' in name_lower:
        if tp_size > 1:
            # If TP is enabled, some AllReduce might be for TP
            # For now, still classify as DP
            return 'DP'
        return 'DP'
    
    # 6. Broadcast -> OTHER (usually initialization)
    if 'broadcast' in name_lower:
        return 'OTHER'
    
    # 7. Default -> OTHER
        return 'OTHER'

def analyze_phase_windows(nsys_rep_file):
    """
    Analyze time windows between different parallelism phases
    
    METRIC 2 Implementation:
      1. Load all NCCL kernels sorted by start time
      2. Categorize each kernel by parallelism type (DP/TP/PP/EP/OTHER)
      3. Group consecutive same-category events into phases
      4. Calculate window time between adjacent phases:
         window = start(next_phase) - end(current_phase)
      5. Aggregate statistics by transition type (e.g., DP→PP, PP→DP)
    
    Returns:
        dict: Window statistics by phase transition
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing phase windows in {nsys_rep_file}...")
    
    # Export to SQLite
    sqlite_file = export_to_sqlite(nsys_rep_file)
    if not sqlite_file:
        print("  Failed to export SQLite, cannot analyze phase windows")
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query NCCL kernels with timeline
        query = """
        SELECT k.start, k.end, s.value as name, k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE '%nccl%'
        ORDER BY k.start
        """
        
        try:
            cursor.execute(query)
            kernels = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"  SQL query failed: {e}")
            conn.close()
            os.remove(sqlite_file)
            return None
        
        if not kernels:
            print("  No NCCL kernels found in trace")
            conn.close()
            os.remove(sqlite_file)
            return None
        
        conn.close()
        os.remove(sqlite_file)
        
        # Categorize events using the corrected function
        events = []
        category_counts = defaultdict(int)
        
        for start, end, name, stream in kernels:
            category = categorize_nccl_kernel(name)  # Use corrected function
            category_counts[category] += 1
            events.append({
                'start': start,
                'end': end,
                'category': category,
                'stream': stream,
                'name': name
            })
        
        print(f"  Found {len(events)} NCCL events")
        print(f"  Category distribution: {dict(category_counts)}")
        
        # Group consecutive events of same category into phases
        phases = []
        if events:
            current_phase = {
                'category': events[0]['category'],
                'start': events[0]['start'],
                'end': events[0]['end'],
                'event_count': 1
            }
            
            for event in events[1:]:
                if event['category'] == current_phase['category']:
                    # Extend current phase
                    current_phase['end'] = max(current_phase['end'], event['end'])
                    current_phase['event_count'] += 1
                else:
                    # Save current phase and start new one
                    phases.append(current_phase)
                    current_phase = {
                        'category': event['category'],
                        'start': event['start'],
                        'end': event['end'],
                        'event_count': 1
                    }
            
            # Add last phase
            phases.append(current_phase)
        
        print(f"  Grouped into {len(phases)} phases")
        
        # Calculate windows between phases
        windows = defaultdict(list)
        positive_windows = 0  # idle gaps
        negative_windows = 0  # overlaps
        
        for i in range(len(phases) - 1):
            phase1 = phases[i]
            phase2 = phases[i + 1]
            
            # Window time = start of next phase - end of current phase
            window_time_ns = phase2['start'] - phase1['end']
            
            if window_time_ns > 0:
                positive_windows += 1
            else:
                negative_windows += 1
            
            # Categorize by transition type
            transition = f"{phase1['category']}->{phase2['category']}"
            windows[transition].append(window_time_ns)
        
        # Calculate statistics for each transition type
        stats = {
            'transitions': {},
            'total_windows': sum(len(w) for w in windows.values()),
            'positive_windows': positive_windows,  # idle gaps
            'negative_windows': negative_windows,  # overlaps
            'num_phases': len(phases),
            'category_counts': dict(category_counts)
        }
        
        for transition, window_list in windows.items():
            if window_list:
                window_list_ms = [w / 1e6 for w in window_list]  # Convert to ms
                
                # Count positive (idle) vs negative (overlap) windows
                positive = sum(1 for w in window_list_ms if w > 0)
                negative = sum(1 for w in window_list_ms if w < 0)
                
                stats['transitions'][transition] = {
                    'count': len(window_list),
                    'mean_window_ms': np.mean(window_list_ms),
                    'p50_window_ms': np.percentile(window_list_ms, 50),
                    'p95_window_ms': np.percentile(window_list_ms, 95),
                    'min_window_ms': np.min(window_list_ms),
                    'max_window_ms': np.max(window_list_ms),
                    'positive_count': positive,  # idle gaps
                    'negative_count': negative,  # overlaps
                    # For backward compatibility
                    'mean_ms': np.mean(window_list_ms),
                    'p50_ms': np.percentile(window_list_ms, 50),
                    'p95_ms': np.percentile(window_list_ms, 95),
                    'min_ms': np.min(window_list_ms),
                    'max_ms': np.max(window_list_ms)
                }
        
        return stats
        
    except Exception as e:
        print(f"  Error analyzing phase windows: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return None

def metric_cal(directory):
    """
    Calculate phase window metrics (METRIC 2)
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        dict: Phase window statistics
    """
    # Find nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return {}
    
    # Analyze first nsys file
    for filename in os.listdir(nsys_dir):
        if filename.endswith(".nsys-rep"):
            nsys_file = os.path.join(nsys_dir, filename)
            stats = analyze_phase_windows(nsys_file)
            
            if stats:
                print("\n" + "="*70)
                print("METRIC 2: Parallelism-Phase Window Time Analysis")
                print("="*70)
                print(f"Total phases: {stats.get('num_phases', 'N/A')}")
                print(f"Total transitions: {stats['total_windows']}")
                print(f"  - Idle gaps (window > 0): {stats.get('positive_windows', 'N/A')}")
                print(f"  - Overlaps (window < 0): {stats.get('negative_windows', 'N/A')}")
                
                if stats.get('category_counts'):
                    print(f"\nEvent distribution by category:")
                    for cat, count in sorted(stats['category_counts'].items()):
                        print(f"  {cat}: {count} events")
                
                print(f"\n{'Transition':<15} {'Count':>8} {'Mean':>10} {'P50':>10} {'P95':>10} {'Idle':>6} {'Overlap':>8}")
                print("-" * 75)
                
                for transition, data in sorted(stats['transitions'].items()):
                    mean = data['mean_window_ms']
                    p50 = data['p50_window_ms']
                    p95 = data['p95_window_ms']
                    pos = data.get('positive_count', 0)
                    neg = data.get('negative_count', 0)
                    
                    print(f"{transition:<15} {data['count']:>8} {mean:>10.2f} {p50:>10.2f} {p95:>10.2f} {pos:>6} {neg:>8}")
                
                print("-" * 75)
                print("\nInterpretation:")
                print("  - window > 0 (Idle): Gap between phases, GPU may be idle or computing")
                print("  - window < 0 (Overlap): Phases overlap, good for hiding communication latency")
                print("="*70)
                
                return stats
            break
    
    return {}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        
        # Print METRIC 2 summary
        if result.get('transitions'):
            print(f"\n=== METRIC 2 Summary ===")
            for transition, data in sorted(result['transitions'].items()):
                print(f"{transition}: mean_window={data['mean_window_ms']:.2f}ms, count={data['count']}")
    else:
        print("Usage: python phase_window_analyzer.py <nsys_directory>")

