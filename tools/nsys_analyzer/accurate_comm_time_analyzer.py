#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accurate Communication Time Analyzer
Uses timeline analysis to compute true wall-clock communication percentage
"""

import os
import sqlite3
from collections import defaultdict

def analyze_accurate_comm_time(nsys_rep_file):
    """
    Analyze GPU timeline to compute accurate communication time percentage
    
    Method:
    1. Load all GPU kernels with start/end times
    2. Build timeline events
    3. Sweep through timeline to calculate time spent in comm vs compute
    
    Returns:
        dict: Accurate timing statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing accurate communication time in {nsys_rep_file}...")
    
    # Check for existing SQLite export
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    
    if not os.path.exists(sqlite_file):
        print(f"  SQLite file not found: {sqlite_file}")
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query all CUDA kernels with timing information
        query = """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """
        
        cursor.execute(query)
        kernels = cursor.fetchall()
        conn.close()
        
        if not kernels:
            print("  No kernel data found")
            return None
        
        print(f"  Found {len(kernels)} GPU kernels")
        
        # Categorize kernels and build timeline events
        timeline_events = []
        
        for start, end, name in kernels:
            name_str = str(name) if name else ""
            
            if 'nccl' in name_str.lower():
                kernel_type = 'comm'
            else:
                kernel_type = 'compute'
            
            timeline_events.append({
                'time': start,
                'type': 'start',
                'category': kernel_type,
                'name': name_str
            })
            timeline_events.append({
                'time': end,
                'type': 'end',
                'category': kernel_type,
                'name': name_str
            })
        
        # Sort events by time
        timeline_events.sort(key=lambda x: x['time'])
        
        # Sweep through timeline and calculate time in each state
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
            else:  # end
                if event['category'] == 'comm':
                    comm_active_count -= 1
                else:
                    compute_active_count -= 1
            
            last_time = current_time
        
        # Convert from nanoseconds to milliseconds
        total_time_ms = total_time / 1e6
        comm_only_time_ms = comm_only_time / 1e6
        compute_only_time_ms = compute_only_time / 1e6
        overlap_time_ms = overlap_time / 1e6
        idle_time_ms = idle_time / 1e6
        
        # Calculate effective communication time (comm_only + overlap)
        effective_comm_time_ms = comm_only_time_ms + overlap_time_ms
        effective_compute_time_ms = compute_only_time_ms + overlap_time_ms
        
        # Calculate percentages
        comm_percentage = (effective_comm_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        compute_percentage = (effective_compute_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        overlap_percentage = (overlap_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        idle_percentage = (idle_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        
        results = {
            'total_time_ms': total_time_ms,
            'comm_only_time_ms': comm_only_time_ms,
            'compute_only_time_ms': compute_only_time_ms,
            'overlap_time_ms': overlap_time_ms,
            'idle_time_ms': idle_time_ms,
            'effective_comm_time_ms': effective_comm_time_ms,
            'effective_compute_time_ms': effective_compute_time_ms,
            'comm_percentage': comm_percentage,
            'compute_percentage': compute_percentage,
            'overlap_percentage': overlap_percentage,
            'idle_percentage': idle_percentage,
            'num_kernels': len(kernels)
        }
        
        print(f"\n  Timeline Analysis Results:")
        print(f"    Total GPU time: {total_time_ms:.2f} ms")
        print(f"    Communication only: {comm_only_time_ms:.2f} ms ({comm_only_time_ms/total_time_ms*100:.1f}%)")
        print(f"    Compute only: {compute_only_time_ms:.2f} ms ({compute_only_time_ms/total_time_ms*100:.1f}%)")
        print(f"    Overlap: {overlap_time_ms:.2f} ms ({overlap_percentage:.1f}%)")
        print(f"    Idle: {idle_time_ms:.2f} ms ({idle_percentage:.1f}%)")
        print(f"    ---")
        print(f"    Effective comm time: {effective_comm_time_ms:.2f} ms ({comm_percentage:.1f}%)")
        print(f"    Effective compute time: {effective_compute_time_ms:.2f} ms ({compute_percentage:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        nsys_file = sys.argv[1]
        result = analyze_accurate_comm_time(nsys_file)
        if result:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"Communication percentage: {result['comm_percentage']:.1f}%")
            print(f"Compute percentage: {result['compute_percentage']:.1f}%")
            print(f"Overlap percentage: {result['overlap_percentage']:.1f}%")
            print(f"Idle percentage: {result['idle_percentage']:.1f}%")
    else:
        print("Usage: python accurate_comm_time_analyzer.py <nsys-rep-file>")

