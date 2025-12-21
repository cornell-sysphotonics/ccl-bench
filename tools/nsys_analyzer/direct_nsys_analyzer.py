#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct analysis of nsys-rep files using nsys stats command
More reliable than parsing JSON exports
"""

import subprocess
import re
import os
from collections import defaultdict

def analyze_nsys_rep(nsys_rep_file):
    """
    Analyze nsys-rep file directly using nsys stats
    
    Returns:
        dict: Communication statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing {nsys_rep_file}...")
    
    # Run nsys stats to get CUDA kernel statistics
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", nsys_rep_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        output = result.stdout.decode('utf-8')
        
        # Parse output for NCCL kernels
        nccl_stats = {
            "total_calls": 0,
            "call_types": defaultdict(int),
            "total_time_ns": 0
        }
        
        lines = output.split('\n')
        in_table = False
        
        for line in lines:
            # Look for NCCL kernel entries
            if 'nccl' in line.lower():
                # Parse table format: Time(%) Total Time(ns) Instances Avg(ns) Med(ns) Min(ns) Max(ns) StdDev(ns) Name
                parts = line.split()
                if len(parts) >= 9:  # Ensure we have all fields
                    try:
                        # Extract instances (3rd column, 0-indexed as 2)
                        instances_str = parts[2].replace(',', '')
                        if instances_str.isdigit():
                            instances = int(instances_str)
                            nccl_stats["total_calls"] += instances
                            
                            # Extract full kernel name (everything from column 8 onwards)
                            # Join all parts from index 8 to end to get complete name
                            name = ' '.join(parts[8:])
                            
                            # Categorize by type - check the full line for better matching
                            line_lower = line.lower()
                            if "allreduce" in line_lower:
                                nccl_stats["call_types"]["AllReduce"] += instances
                            elif "reducescatter" in line_lower or "reduce_scatter" in line_lower:
                                nccl_stats["call_types"]["ReduceScatter"] += instances
                            elif "allgather" in line_lower or "all_gather" in line_lower:
                                nccl_stats["call_types"]["AllGather"] += instances
                            elif "sendrecv" in line_lower or "send" in line_lower or "recv" in line_lower:
                                nccl_stats["call_types"]["SendRecv"] += instances
                            elif "broadcast" in line_lower:
                                nccl_stats["call_types"]["Broadcast"] += instances
                            elif "alltoall" in line_lower or "all_to_all" in line_lower:
                                nccl_stats["call_types"]["AllToAll"] += instances
                            else:
                                nccl_stats["call_types"]["Other"] += instances
                    except (ValueError, IndexError):
                        continue
        
        return nccl_stats
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {nsys_rep_file}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def analyze_trace_directory(nsys_path):
    """
    Analyze all nsys-rep files in a directory, or analyze a single nsys-rep file
    
    Args:
        nsys_path: Directory containing nsys-rep files, or single nsys-rep file path
    
    Returns:
        dict: Aggregated statistics
    """
    results = {
        "total_nccl_calls": 0,
        "nccl_call_types": defaultdict(int),
        "files_analyzed": []
    }
    
    # 1. Check if input is single file or directory
    nsys_files = []
    if os.path.isfile(nsys_path):
        # Single file mode
        if nsys_path.endswith(".nsys-rep"):
            nsys_files.append(nsys_path)
        else:
            print(f"Error: File is not a .nsys-rep file: {nsys_path}")
            return results
    elif os.path.isdir(nsys_path):
        # Directory mode: find all nsys-rep files
        for filename in os.listdir(nsys_path):
            if filename.endswith(".nsys-rep"):
                nsys_files.append(os.path.join(nsys_path, filename))
    else:
        print(f"Error: Invalid path: {nsys_path}")
        return results
    
    if not nsys_files:
        print(f"No .nsys-rep files found in {nsys_path}")
        return results
    
    print(f"Found {len(nsys_files)} nsys-rep file(s)\n")
    
    # 2. Analyze each file
    for nsys_file in nsys_files:
        stats = analyze_nsys_rep(nsys_file)
        if stats:
            results["total_nccl_calls"] += stats["total_calls"]
            for call_type, count in stats["call_types"].items():
                results["nccl_call_types"][call_type] += count
            results["files_analyzed"].append(os.path.basename(nsys_file))
            print(f"  Found {stats['total_calls']} NCCL calls\n")
    
    return results

def metric_cal(directory):
    """
    Calculate NCCL communication metrics
    Compatible with CCL-Bench interface
    
    Args:
        directory: Can be either trace_collection directory or nsys directory
    
    Returns:
        int: Total number of NCCL calls
    """
    # Check if this is a trace_collection directory or nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        # This is a trace_collection directory, look for nsys files
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        # This directory contains nsys-rep files
        nsys_dir = directory
    else:
        print(f"No nsys-rep files found. Please provide nsys directory.")
        print(f"Usage: python direct_nsys_analyzer.py <nsys_directory>")
        return 0
    
    results = analyze_trace_directory(nsys_dir)
    
    print("\n" + "="*60)
    print("NCCL Communication Analysis Results (from nsys-rep)")
    print("="*60)
    print(f"Total NCCL calls: {results['total_nccl_calls']}")
    print(f"Files analyzed: {', '.join(results['files_analyzed'])}")
    print("\nCall type breakdown:")
    for call_type, count in sorted(results['nccl_call_types'].items()):
        print(f"  {call_type}: {count}")
    print("="*60)
    
    return results['total_nccl_calls']

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        print(f"\nTotal NCCL calls: {result}")
    else:
        print("Usage: python direct_nsys_analyzer.py <nsys_directory>")

