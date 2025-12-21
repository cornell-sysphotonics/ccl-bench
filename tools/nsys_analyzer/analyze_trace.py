#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Entry Point for Trace Analysis
Analyzes nsys traces and computes all metrics

Usage:
    python analyze_trace.py <nsys_directory_or_file> [options]

Examples:
    # Analyze directory with ALL trace files (results merged into one JSON)
    python analyze_trace.py ../../nsys_2node
    
    # Analyze single trace file
    python analyze_trace.py ../../nsys/trace_2nodes_rank_0.nsys-rep
    
    # Run specific metrics
    python analyze_trace.py ../../nsys --metrics nccl_calls,iteration_time
    
    # Export results
    python analyze_trace.py ../../nsys --output results.json --csv results.csv
    
    # Specify workload name and parallelism config
    python analyze_trace.py ../../nsys --name llama-3.1-8b-1n4g --dp 8 --tp 1 --pp 1 --ep 1
"""

import os
import sys
import json
import argparse
from collections import OrderedDict

# Import all metric analyzers
import direct_nsys_analyzer
import iteration_time_analyzer
import comm_time_breakdown
import comm_compute_overlap
import phase_window_analyzer
import traffic_interval_analyzer

# Define all available metrics
AVAILABLE_METRICS = {
    'nccl_calls': {
        'name': 'NCCL Communication Calls',
        'module': direct_nsys_analyzer,
        'description': 'Count and categorize NCCL communication calls'
    },
    'iteration_time': {
        'name': 'Iteration Time Statistics',
        'module': iteration_time_analyzer,
        'description': 'Calculate mean, P50, P99 iteration times'
    },
    'comm_breakdown': {
        'name': 'Communication Time Breakdown',
        'module': comm_time_breakdown,
        'description': 'Break down time by DP/TP/PP/EP communication'
    },
    'overlap': {
        'name': 'Communication-Computation Overlap',
        'module': comm_compute_overlap,
        'description': 'Measure overlap between comm and compute'
    },
    'phase_windows': {
        'name': 'Parallelism Phase Windows',
        'module': phase_window_analyzer,
        'description': 'Analyze time gaps between parallelism phases'
    },
    'traffic_interval': {
        'name': 'Traffic Interval Analysis',
        'module': traffic_interval_analyzer,
        'description': 'Analyze call intervals and durations for each NCCL operation type'
    }
}

# Global parallelism config (set from command line)
PARALLELISM_CONFIG = {
    'dp_size': 1,
    'tp_size': 1,
    'pp_size': 1,
    'ep_size': 1
}

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_subheader(text):
    """Print formatted subsection header"""
    print("\n" + "-"*70)
    print(text)
    print("-"*70)

def validate_nsys_input(nsys_path):
    """
    Validate input path (can be directory or single file)
    
    Returns:
        tuple: (is_valid, is_file, trace_files)
            - is_valid: whether the input is valid
            - is_file: whether input is a single file
            - trace_files: list of trace files
    """
    if not os.path.exists(nsys_path):
        print(f"Error: Path not found: {nsys_path}")
        return False, False, []
    
    # 1. Check if it's a single .nsys-rep file
    if os.path.isfile(nsys_path):
        if nsys_path.endswith('.nsys-rep'):
            return True, True, [nsys_path]
        else:
            print(f"Error: File is not a .nsys-rep file: {nsys_path}")
            return False, False, []
    
    # 2. Check if it's a directory containing .nsys-rep files
    if os.path.isdir(nsys_path):
        nsys_files = [f for f in os.listdir(nsys_path) if f.endswith('.nsys-rep')]
        if not nsys_files:
            print(f"Error: No .nsys-rep files found in directory: {nsys_path}")
            return False, False, []
        return True, False, nsys_files
    
    print(f"Error: Invalid path: {nsys_path}")
    return False, False, []

def run_metric_for_file(metric_key, trace_file):
    """
    Run a single metric analyzer for a specific trace file
    
    Args:
        metric_key: metric identifier
        trace_file: path to a single .nsys-rep file
    
    Returns:
        dict: metric analysis results
    """
    if metric_key not in AVAILABLE_METRICS:
        print(f"Warning: Unknown metric '{metric_key}'")
        return None
    
    metric_info = AVAILABLE_METRICS[metric_key]
    
    try:
        # Run different metric analyses
        if metric_key == 'nccl_calls':
            results = metric_info['module'].analyze_trace_directory(trace_file)
        elif metric_key == 'iteration_time':
                results = metric_info['module'].analyze_iteration_time(trace_file)
        elif metric_key == 'comm_breakdown':
            breakdown = metric_info['module'].analyze_comm_breakdown(trace_file)
            results = breakdown
        elif metric_key == 'overlap':
            overlap_stats = metric_info['module'].analyze_overlap(trace_file)
            if overlap_stats:
                # Calculate overlap percentage
                if overlap_stats.get("total_nccl_time_ns", 0) > 0:
                    overlap_pct = (overlap_stats["overlap_time_ns"] / overlap_stats["total_nccl_time_ns"]) * 100
                else:
                    overlap_pct = 0.0
                results = {
                    'overlap_percentage': overlap_pct,
                    'total_nccl_time_ns': overlap_stats.get("total_nccl_time_ns", 0),
                    'total_compute_time_ns': overlap_stats.get("total_compute_time_ns", 0),
                    'overlap_time_ns': overlap_stats.get("overlap_time_ns", 0),
                    'is_estimated': overlap_stats.get("is_estimated", False)
                }
            else:
                results = None
        elif metric_key == 'phase_windows':
            results = metric_info['module'].analyze_phase_windows(trace_file)
        elif metric_key == 'traffic_interval':
            results = metric_info['module'].analyze_traffic_intervals(trace_file)
        else:
            results = None
        
        return results
        
    except Exception as e:
        print(f"✗ Error running {metric_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_rank_from_filename(filename):
    """
    Extract rank number from trace filename
    
    Examples:
        trace_2nodes_rank_0_50.nsys-rep -> 0
        trace_2nodes_rank_4_50.nsys-rep -> 4
        llama_3.1_8b_trace_50.nsys-rep -> None
    """
    import re
    # Try to match rank_X pattern
    match = re.search(r'rank[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def run_all_metrics_for_file(trace_file, metrics_to_run):
    """
    Run all specified metrics for a single trace file
    
    Args:
        trace_file: path to .nsys-rep file
        metrics_to_run: list of metric keys to run
    
    Returns:
        dict: results for all metrics
    """
    results = OrderedDict()
    
    for metric_key in metrics_to_run:
        metric_info = AVAILABLE_METRICS[metric_key]
        metric_results = run_metric_for_file(metric_key, trace_file)
        results[metric_key] = metric_results
        
        if metric_results:
            print(f"  ✓ {metric_key}")
        else:
            print(f"  ✗ {metric_key}")
    
    return results


def aggregate_results(per_rank_results):
    """
    Aggregate results from multiple ranks into summary statistics
    
    Args:
        per_rank_results: dict mapping rank_id -> metric results
    
    Returns:
        dict: aggregated statistics
    """
    aggregated = OrderedDict()
    
    # Collect iteration times across all ranks
    all_iteration_times = []
    for rank_id, rank_data in per_rank_results.items():
        iter_data = rank_data.get('iteration_time')
        if iter_data and iter_data.get('iteration_times_ms'):
            all_iteration_times.extend(iter_data['iteration_times_ms'])
    
    if all_iteration_times:
        import numpy as np
        arr = np.array(all_iteration_times)
        aggregated['iteration_time'] = {
            'num_iterations_total': len(arr),
            'num_ranks': len(per_rank_results),
            'avg_iteration_time_ms': float(np.mean(arr)),
            'std_iteration_time_ms': float(np.std(arr)),
            'min_iteration_time_ms': float(np.min(arr)),
            'max_iteration_time_ms': float(np.max(arr)),
            'p50_iteration_time_ms': float(np.percentile(arr, 50)),
            'p99_iteration_time_ms': float(np.percentile(arr, 99))
        }
    
    # Collect overlap ratios
    overlap_ratios = []
    for rank_id, rank_data in per_rank_results.items():
        overlap_data = rank_data.get('overlap')
        if overlap_data and 'overlap_percentage' in overlap_data:
            overlap_ratios.append(overlap_data['overlap_percentage'])
    
    if overlap_ratios:
        aggregated['overlap'] = {
            'num_ranks': len(overlap_ratios),
            'avg_overlap_percentage': sum(overlap_ratios) / len(overlap_ratios),
            'min_overlap_percentage': min(overlap_ratios),
            'max_overlap_percentage': max(overlap_ratios)
        }
    
    return aggregated

def export_results(results, output_file):
    """Export results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results exported to: {output_file}")
    except Exception as e:
        print(f"\n✗ Failed to export JSON: {e}")

def export_csv(results, csv_file):
    """Export summary results to CSV"""
    import csv
    
    try:
        # Safely extract key metrics for CSV
        nccl_calls = results.get('nccl_calls') or {}
        iteration_time = results.get('iteration_time') or {}
        comm_breakdown = results.get('comm_breakdown') or {}
        overlap = results.get('overlap') or {}
        
        # Get communication percentage from timeline analysis if available
        comm_pct = 'N/A'
        if comm_breakdown.get('timeline_analysis'):
            comm_pct = comm_breakdown['timeline_analysis'].get('comm_percentage', 'N/A')
        elif 'communication_percentage' in comm_breakdown:
            comm_pct = comm_breakdown.get('communication_percentage', 'N/A')
        
        row = {
            'workload_name': results.get('workload_name', 'unknown'),
            'num_traces': results.get('num_traces', 0),
            'nccl_total_calls': nccl_calls.get('total_nccl_calls', 'N/A'),
            'avg_iteration_time_ms': iteration_time.get('avg_iteration_time_ms', 'N/A'),
            'p99_iteration_time_ms': iteration_time.get('p99_iteration_time_ms', 'N/A'),
            'communication_pct': comm_pct,
            'overlap_pct': overlap.get('overlap_percentage', 'N/A') if isinstance(overlap, dict) else 'N/A'
        }
        
        # Check if file exists
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        print(f"✓ Results appended to: {csv_file}")
        
    except Exception as e:
        print(f"✗ Failed to export CSV: {e}")

def main():
    global PARALLELISM_CONFIG
    
    parser = argparse.ArgumentParser(
        description='Unified trace analyzer for nsys profiling data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all trace files in directory (results merged into one JSON)
  python analyze_trace.py nsys_2node/ -o results.json
  
  # Analyze single trace file
  python analyze_trace.py ../../nsys/trace_2nodes_rank_0.nsys-rep
  
  # Run specific metrics only
  python analyze_trace.py ../../nsys --metrics nccl_calls,iteration_time
  
  # Export analysis results
  python analyze_trace.py ../../nsys --output results.json --csv summary.csv
  
  # Specify workload name and parallelism config
  python analyze_trace.py ../../nsys --name llama-3.1-8b --dp 8 --tp 1 --pp 1 --ep 1

Available metrics:
  nccl_calls        - NCCL communication call statistics
  iteration_time    - Iteration timing statistics (mean, P99)
  comm_breakdown    - Communication time breakdown by type
  overlap           - Communication-computation overlap analysis
  phase_windows     - Parallelism phase window analysis
  traffic_interval  - Call intervals and durations per operation type
        """
    )
    
    parser.add_argument('nsys_path', help='Directory containing nsys-rep files, or single nsys-rep file path')
    parser.add_argument('--name', default='unknown', help='Workload name for identification')
    parser.add_argument('--metrics', help='Comma-separated list of metrics (default: all)')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--csv', help='Output CSV file path (append mode)')
    parser.add_argument('--list', action='store_true', help='List available metrics and exit')
    
    # Parallelism configuration
    parser.add_argument('--dp', type=int, default=1, help='Data parallel size (default: 1)')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size (default: 1)')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline parallel size (default: 1)')
    parser.add_argument('--ep', type=int, default=1, help='Expert parallel size (default: 1)')
    
    args = parser.parse_args()
    
    # Set global parallelism config
    PARALLELISM_CONFIG['dp_size'] = args.dp
    PARALLELISM_CONFIG['tp_size'] = args.tp
    PARALLELISM_CONFIG['pp_size'] = args.pp
    PARALLELISM_CONFIG['ep_size'] = args.ep
    
    # List metrics and exit
    if args.list:
        print("Available Metrics:")
        print("-" * 70)
        for key, info in AVAILABLE_METRICS.items():
            print(f"  {key:15s} - {info['description']}")
        return 0
    
    # Validate input path
    is_valid, is_single_file, trace_files = validate_nsys_input(args.nsys_path)
    if not is_valid:
        return 1
    
    # Determine which metrics to run
    if args.metrics:
        metrics_to_run = [m.strip() for m in args.metrics.split(',')]
        # Validate metric names
        invalid = [m for m in metrics_to_run if m not in AVAILABLE_METRICS]
        if invalid:
            print(f"Error: Unknown metrics: {', '.join(invalid)}")
            print(f"Use --list to see available metrics")
            return 1
    else:
        metrics_to_run = list(AVAILABLE_METRICS.keys())
    
    # Print analysis header
    print_header(f"Trace Analysis: {args.name}")
    if is_single_file:
        print(f"Mode: Single file")
        print(f"Trace file: {os.path.basename(args.nsys_path)}")
    else:
        print(f"Mode: Multi-file (results merged)")
        print(f"Directory: {args.nsys_path}")
        print(f"Trace files ({len(trace_files)}):")
        for tf in trace_files:
            print(f"  - {tf}")
    print(f"Parallelism config: DP={args.dp}, TP={args.tp}, PP={args.pp}, EP={args.ep}")
    print(f"Metrics to run: {', '.join(metrics_to_run)}")
    
    # Build results structure
    results = OrderedDict()
    results['workload_name'] = args.name
    results['nsys_path'] = args.nsys_path
    results['is_single_file'] = is_single_file
    results['num_traces'] = 1 if is_single_file else len(trace_files)
    results['trace_files'] = [os.path.basename(args.nsys_path)] if is_single_file else trace_files
    results['parallelism_config'] = {
        'dp_size': args.dp,
        'tp_size': args.tp,
        'pp_size': args.pp,
        'ep_size': args.ep
    }
    
    if is_single_file:
        # Single file mode: run all metrics for the single file
        print_subheader(f"Analyzing: {os.path.basename(args.nsys_path)}")
        file_results = run_all_metrics_for_file(args.nsys_path, metrics_to_run)
        
        # Put results at top level for backward compatibility
        for metric_key, metric_result in file_results.items():
            results[metric_key] = metric_result
    else:
        # Multi-file mode: analyze each trace file and merge results
        per_rank_results = OrderedDict()
        
        for trace_filename in sorted(trace_files):
            trace_path = os.path.join(args.nsys_path, trace_filename)
            rank_id = extract_rank_from_filename(trace_filename)
            
            if rank_id is not None:
                rank_key = f"rank_{rank_id}"
            else:
                # Use filename as key if no rank found
                rank_key = os.path.splitext(trace_filename)[0]
            
            print_subheader(f"Analyzing: {trace_filename} ({rank_key})")
            file_results = run_all_metrics_for_file(trace_path, metrics_to_run)
            
            # Store per-rank results
            per_rank_results[rank_key] = {
                'trace_file': trace_filename,
                **file_results
            }
        
        # Add per-rank results to output
        results['per_rank_results'] = per_rank_results
        
        # Calculate aggregated statistics across all ranks
        print_subheader("Computing aggregated statistics")
        aggregated = aggregate_results(per_rank_results)
        results['aggregated'] = aggregated
        
        print(f"  ✓ Aggregated {len(per_rank_results)} ranks")
    
    # Print summary
    print_header("Analysis Summary")
    print(f"Workload: {args.name}")
    if is_single_file:
        print(f"Traces analyzed: 1")
    else:
        print(f"Traces analyzed: {len(trace_files)} (merged into one output)")
        print(f"Ranks: {', '.join(sorted(results.get('per_rank_results', {}).keys()))}")
    
    # Show aggregated summary for multi-file mode
    if not is_single_file and results.get('aggregated'):
        agg = results['aggregated']
        print("\nAggregated Statistics:")
        if agg.get('iteration_time'):
            it = agg['iteration_time']
            print(f"  Iteration Time: avg={it['avg_iteration_time_ms']:.2f}ms, "
                  f"p99={it['p99_iteration_time_ms']:.2f}ms")
        if agg.get('overlap'):
            ov = agg['overlap']
            print(f"  Overlap: avg={ov['avg_overlap_percentage']:.2f}%")
    
    # Export results if requested
    if args.output:
        export_results(results, args.output)
    
    if args.csv:
        export_csv(results, args.csv)
    
    print("\n✓ Analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

