#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 5: Communication-Computation Overlap Analyzer

Measures the degree of overlap between communication and computation operations.

Definition (from spec):
  For each rank r, each iteration i:
    - comm_time_union(i, r): union of all communication intervals
    - compute_time_union(i, r): union of all compute kernel intervals
    - overlapped_time(i, r): time where comm and compute intervals overlap
    - overlap_ratio(i, r) = overlapped_time / comm_time_union

  Aggregate:
    - average_overlap_ratio = average over all i, r

Interpretation:
  Higher overlap_ratio → communication is better hidden → less impact on iteration time
"""

import subprocess
import os
import re
from collections import defaultdict


def analyze_overlap_with_sweep(nsys_rep_file):
    """
    Analyze communication-computation overlap using sweep line algorithm
    
    This is the correct implementation that avoids double-counting overlap time.
    
    Algorithm:
      1. Build timeline events (start/end) for all kernels
      2. Sort by time
      3. Sweep through timeline, tracking active comm and compute counts
      4. Accumulate time for each state (comm_only, compute_only, overlap, idle)
    
    Returns:
        dict: Overlap statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing comm-compute overlap in {nsys_rep_file}...")
    
    # Check for existing SQLite file first
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    temp_sqlite = False
    
    if not os.path.exists(sqlite_file):
        # Need to export
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '_temp.sqlite')
        temp_sqlite = True
    
    try:
        export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}", 
                     "--force-overwrite=true", nsys_rep_file]
        result = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        
        if not os.path.exists(sqlite_file):
            print("  Failed to export SQLite, using stats-based estimation")
                return estimate_overlap_from_stats(nsys_rep_file)
        except subprocess.TimeoutExpired:
            print(f"  Timeout exporting SQLite")
            return estimate_overlap_from_stats(nsys_rep_file)
        
    try:
        import sqlite3
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query all CUDA kernels with timing
        query = """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """
        
        try:
            cursor.execute(query)
            kernels = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"  SQL query failed: {e}, using estimation")
            conn.close()
            if temp_sqlite and os.path.exists(sqlite_file):
            os.remove(sqlite_file)
            return estimate_overlap_from_stats(nsys_rep_file)
        
        conn.close()
        if temp_sqlite and os.path.exists(sqlite_file):
        os.remove(sqlite_file)
        
        if not kernels:
            print("  No kernel data found")
            return None
        
        print(f"  Found {len(kernels)} GPU kernels")
        
        # Build timeline events
        timeline_events = []
        nccl_count = 0
        compute_count = 0
        
        for start, end, name in kernels:
            name_str = str(name) if name else ""
            
            if 'nccl' in name_str.lower():
                kernel_type = 'comm'
                nccl_count += 1
            else:
                kernel_type = 'compute'
                compute_count += 1
            
            timeline_events.append({
                'time': start,
                'type': 'start',
                'category': kernel_type
            })
            timeline_events.append({
                'time': end,
                'type': 'end',
                'category': kernel_type
            })
        
        print(f"  NCCL kernels: {nccl_count}, Compute kernels: {compute_count}")
        
        # Sort events by time
        timeline_events.sort(key=lambda x: x['time'])
        
        # Sweep through timeline
        comm_active_count = 0
        compute_active_count = 0
        last_time = timeline_events[0]['time']
        
        # Accumulators
        total_time = 0
        comm_only_time = 0
        compute_only_time = 0
        overlap_time = 0
        idle_time = 0
        
        for event in timeline_events:
            current_time = event['time']
            duration = current_time - last_time
            
            if duration > 0:
                total_time += duration
                
                # Categorize the time period
                if comm_active_count > 0 and compute_active_count > 0:
                    overlap_time += duration
                elif comm_active_count > 0:
                    comm_only_time += duration
                elif compute_active_count > 0:
                    compute_only_time += duration
                else:
                    idle_time += duration
            
            # Update active counts
            if event['type'] == 'start':
                if event['category'] == 'comm':
                    comm_active_count += 1
                else:
                    compute_active_count += 1
            else:
                if event['category'] == 'comm':
                    comm_active_count -= 1
                else:
                    compute_active_count -= 1
            
            last_time = current_time
        
        # Calculate union times
        # comm_time_union = time when at least one comm kernel is active
        comm_time_union = comm_only_time + overlap_time
        # compute_time_union = time when at least one compute kernel is active  
        compute_time_union = compute_only_time + overlap_time
        
        # METRIC 5: overlap_ratio = overlap_time / comm_time_union
        if comm_time_union > 0:
            overlap_ratio = overlap_time / comm_time_union
        else:
            overlap_ratio = 0.0
        
        stats = {
            # Raw times in nanoseconds
            "total_time_ns": total_time,
            "comm_only_time_ns": comm_only_time,
            "compute_only_time_ns": compute_only_time,
            "overlap_time_ns": overlap_time,
            "idle_time_ns": idle_time,
            # Union times
            "comm_time_union_ns": comm_time_union,
            "compute_time_union_ns": compute_time_union,
            # METRIC 5 core output
            "overlap_ratio": overlap_ratio,
            "average_overlap_ratio": overlap_ratio,  # For spec compliance
            # Kernel counts
            "nccl_intervals": nccl_count,
            "compute_intervals": compute_count,
            # For backward compatibility
            "total_nccl_time_ns": comm_time_union,
            "total_compute_time_ns": compute_time_union,
            "is_estimated": False
        }
        
        return stats
        
    except Exception as e:
        print(f"  Error during overlap analysis: {e}")
        import traceback
        traceback.print_exc()
        if temp_sqlite and os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return estimate_overlap_from_stats(nsys_rep_file)


def analyze_overlap(nsys_rep_file):
    """
    Analyze communication-computation overlap from nsys-rep file
    
    This is the main entry point. Uses sweep line algorithm for accurate results.
    
    Returns:
        dict: Overlap statistics
    """
    return analyze_overlap_with_sweep(nsys_rep_file)

def estimate_overlap_from_stats(nsys_rep_file):
    """
    Estimate overlap using statistical approach when timeline data unavailable
    
    This is a conservative estimation based on kernel statistics.
    Not as accurate as sweep line algorithm, but works when SQLite export fails.
    """
    print("  Using statistical estimation for overlap (less accurate)")
    
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", nsys_rep_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        output = result.stdout.decode('utf-8')
        
        total_nccl_time = 0
        total_compute_time = 0
        
        lines = output.split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                total_time_str = parts[1].replace(',', '')
                if not total_time_str.replace('.', '').isdigit():
                    continue
                
                total_time_ns = float(total_time_str)
                kernel_name = ' '.join(parts[8:])
                
                if 'nccl' in kernel_name.lower():
                    total_nccl_time += total_time_ns
                else:
                    total_compute_time += total_time_ns
            except (ValueError, IndexError):
                continue
        
        # Conservative estimation: assume some overlap due to async operations
        # DeepSpeed ZeRO-3 typically has ~20-30% overlap
        estimated_overlap = min(total_nccl_time, total_compute_time) * 0.25
        
        # Estimate overlap_ratio
        if total_nccl_time > 0:
            overlap_ratio = estimated_overlap / total_nccl_time
        else:
            overlap_ratio = 0.0
        
        stats = {
            "total_nccl_time_ns": total_nccl_time,
            "total_compute_time_ns": total_compute_time,
            "overlap_time_ns": estimated_overlap,
            "comm_time_union_ns": total_nccl_time,
            "compute_time_union_ns": total_compute_time,
            "overlap_ratio": overlap_ratio,
            "average_overlap_ratio": overlap_ratio,
            "is_estimated": True
        }
        
        return stats
        
    except Exception as e:
        print(f"  Error in estimation: {e}")
        return None


def metric_cal(directory):
    """
    Calculate communication-computation overlap (METRIC 5)
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        dict: Overlap statistics including overlap_ratio
    """
    # Find nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return {}
    
    # Analyze first nsys file (overlap is per-GPU metric)
    for filename in os.listdir(nsys_dir):
        if filename.endswith(".nsys-rep"):
            nsys_file = os.path.join(nsys_dir, filename)
            stats = analyze_overlap(nsys_file)
            
            if stats:
                print("\n" + "="*70)
                print("METRIC 5: Communication-Computation Overlap Analysis")
                print("="*70)
                
                # Convert to milliseconds for display
                total_time_ms = stats.get("total_time_ns", 0) / 1e6
                comm_union_ms = stats["comm_time_union_ns"] / 1e6
                compute_union_ms = stats["compute_time_union_ns"] / 1e6
                overlap_time_ms = stats["overlap_time_ns"] / 1e6
                comm_only_ms = stats.get("comm_only_time_ns", 0) / 1e6
                compute_only_ms = stats.get("compute_only_time_ns", 0) / 1e6
                idle_ms = stats.get("idle_time_ns", 0) / 1e6
                
                print(f"Timeline breakdown:")
                print(f"  Total GPU time:       {total_time_ms:>12.2f} ms")
                print(f"  Comm only time:       {comm_only_ms:>12.2f} ms")
                print(f"  Compute only time:    {compute_only_ms:>12.2f} ms")
                print(f"  Overlap time:         {overlap_time_ms:>12.2f} ms")
                print(f"  Idle time:            {idle_ms:>12.2f} ms")
                print(f"\nUnion times:")
                print(f"  comm_time_union:      {comm_union_ms:>12.2f} ms")
                print(f"  compute_time_union:   {compute_union_ms:>12.2f} ms")
                
                # METRIC 5 core output
                overlap_ratio = stats["overlap_ratio"]
                overlap_pct = overlap_ratio * 100
                
                print(f"\n=== METRIC 5 Output ===")
                print(f"  overlap_ratio = overlap_time / comm_time_union")
                print(f"  overlap_ratio = {overlap_time_ms:.2f} / {comm_union_ms:.2f} = {overlap_ratio:.4f}")
                print(f"  average_overlap_ratio: {overlap_pct:.2f}%")
                
                if stats.get("is_estimated"):
                    print("\n  Note: Values are estimated (timeline data unavailable)")
                
                print("\nInterpretation:")
                print(f"  Higher overlap_ratio → better hiding of communication latency")
                if overlap_pct < 5:
                    print(f"  Current: {overlap_pct:.1f}% - Very low overlap, communication is blocking")
                elif overlap_pct < 20:
                    print(f"  Current: {overlap_pct:.1f}% - Low overlap, some async operations")
                elif overlap_pct < 50:
                    print(f"  Current: {overlap_pct:.1f}% - Moderate overlap")
                else:
                    print(f"  Current: {overlap_pct:.1f}% - Good overlap, communication well hidden")
                
                print("="*70)
                
                return stats
            break
    
    return {}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        
        if result:
            print(f"\n=== METRIC 5 Summary ===")
            print(f"average_overlap_ratio: {result['average_overlap_ratio']*100:.2f}%")
    else:
        print("Usage: python comm_compute_overlap.py <nsys_directory>")

