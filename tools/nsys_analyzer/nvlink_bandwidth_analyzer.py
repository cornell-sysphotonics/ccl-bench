#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVLink Bandwidth Utilization Analyzer

Reads nsys profiles (sqlite or nsys-rep files) and calculates NVLink bandwidth
utilization from GPU metrics, then generates visualization plots.

NVLink Metrics (from nsys GPU_METRICS table):
  - metricId 20: NVLink RX Requests Protocol Data [Throughput %]
  - metricId 21: NVLink RX Requests User Data [Throughput %]
  - metricId 22: NVLink RX Responses Protocol Data [Throughput %]
  - metricId 23: NVLink RX Responses User Data [Throughput %]
  - metricId 24: NVLink TX Requests Protocol Data [Throughput %]
  - metricId 25: NVLink TX Requests User Data [Throughput %]
  - metricId 26: NVLink TX Responses Protocol Data [Throughput %]
  - metricId 27: NVLink TX Responses User Data [Throughput %]

The values are throughput percentages (0-100).

Usage:
    python nvlink_bandwidth_analyzer.py <sqlite_file_or_nsys_rep> [--output output.png]
    python nvlink_bandwidth_analyzer.py $PSCRATCH/datas/llama_bandwidth_fixed.sqlite
"""

import sqlite3
import subprocess
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# NVLink metric IDs and their names
NVLINK_METRICS = {
    20: 'NVLink RX Requests Protocol Data',
    21: 'NVLink RX Requests User Data',
    22: 'NVLink RX Responses Protocol Data',
    23: 'NVLink RX Responses User Data',
    24: 'NVLink TX Requests Protocol Data',
    25: 'NVLink TX Requests User Data',
    26: 'NVLink TX Responses Protocol Data',
    27: 'NVLink TX Responses User Data',
}

# Simplified metric categories
NVLINK_CATEGORIES = {
    'RX Protocol': [20, 22],
    'RX User Data': [21, 23],
    'TX Protocol': [24, 26],
    'TX User Data': [25, 27],
}

# Aggregate categories
NVLINK_AGGREGATE = {
    'Total RX': [20, 21, 22, 23],
    'Total TX': [24, 25, 26, 27],
    'Total User Data': [21, 23, 25, 27],
    'Total Protocol': [20, 22, 24, 26],
}


def export_nsys_to_sqlite(nsys_rep_file):
    """Export nsys-rep file to SQLite format"""
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '_export.sqlite')
    
    if os.path.exists(sqlite_file):
        print(f"  Using existing SQLite export: {sqlite_file}")
        return sqlite_file
    
    try:
        print(f"  Exporting {nsys_rep_file} to SQLite...")
        export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}", 
                     "--force-overwrite=true", nsys_rep_file]
        result = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        
        if os.path.exists(sqlite_file):
            print(f"  Export successful: {sqlite_file}")
            return sqlite_file
        else:
            print(f"  Export failed: {result.stderr.decode()}")
            return None
    except Exception as e:
        print(f"  Error exporting to SQLite: {e}")
        return None


def load_nvlink_metrics(sqlite_file):
    """
    Load NVLink metrics from SQLite database
    
    Returns:
        dict: {
            'timestamps': np.array of timestamps (ns),
            'metrics': {metricId: np.array of values},
            'gpus': {typeId: gpu_index},
            'gpu_metrics': {typeId: {metricId: np.array}}
        }
    """
    if not os.path.exists(sqlite_file):
        print(f"File not found: {sqlite_file}")
        return None
    
    print(f"Loading NVLink metrics from {sqlite_file}...")
    
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    # Check if GPU_METRICS table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='GPU_METRICS'")
    if not cursor.fetchone():
        print("  GPU_METRICS table not found. This profile may not have GPU metrics enabled.")
        conn.close()
        return None
    
    # Get all unique GPUs (typeId)
    cursor.execute("SELECT DISTINCT typeId FROM GPU_METRICS ORDER BY typeId")
    gpu_types = [row[0] for row in cursor.fetchall()]
    print(f"  Found {len(gpu_types)} GPU(s)")
    
    # Query NVLink metrics for all GPUs
    query = """
    SELECT timestamp, typeId, metricId, value
    FROM GPU_METRICS
    WHERE metricId BETWEEN 20 AND 27
    ORDER BY timestamp, typeId, metricId
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("  No NVLink metrics found in profile")
        return None
    
    print(f"  Found {len(rows)} NVLink metric samples")
    
    # Organize data by GPU and metric
    data = {
        'timestamps': set(),
        'gpus': {gid: idx for idx, gid in enumerate(gpu_types)},
        'gpu_metrics': defaultdict(lambda: defaultdict(list)),
        'timestamp_data': defaultdict(lambda: defaultdict(dict))
    }
    
    for timestamp, typeId, metricId, value in rows:
        data['timestamps'].add(timestamp)
        data['gpu_metrics'][typeId][metricId].append((timestamp, value))
        data['timestamp_data'][timestamp][typeId][metricId] = value
    
    # Convert to sorted arrays
    data['timestamps'] = np.array(sorted(data['timestamps']))
    
    # Convert metric lists to arrays
    for typeId in data['gpu_metrics']:
        for metricId in data['gpu_metrics'][typeId]:
            data['gpu_metrics'][typeId][metricId] = np.array(data['gpu_metrics'][typeId][metricId])
    
    return data


def calculate_utilization_stats(data):
    """
    Calculate NVLink bandwidth utilization statistics
    
    Returns:
        dict: Statistics for each metric and aggregate categories
    """
    if not data:
        return None
    
    stats = {
        'per_gpu': {},
        'aggregate': {},
        'time_series': {}
    }
    
    # Calculate stats per GPU
    for typeId, gpu_idx in data['gpus'].items():
        gpu_stats = {}
        
        for metricId, metric_name in NVLINK_METRICS.items():
            if metricId in data['gpu_metrics'][typeId]:
                values = data['gpu_metrics'][typeId][metricId][:, 1]  # value column
                gpu_stats[metric_name] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'std': np.std(values),
                    'samples': len(values)
                }
        
        # Calculate aggregate categories
        for cat_name, metric_ids in NVLINK_AGGREGATE.items():
            cat_values = []
            for mid in metric_ids:
                if mid in data['gpu_metrics'][typeId]:
                    cat_values.extend(data['gpu_metrics'][typeId][mid][:, 1])
            if cat_values:
                gpu_stats[cat_name] = {
                    'mean': np.mean(cat_values),
                    'max': np.max(cat_values),
                    'p50': np.percentile(cat_values, 50),
                    'p95': np.percentile(cat_values, 95),
                }
        
        stats['per_gpu'][f'GPU_{gpu_idx}'] = gpu_stats
    
    # Calculate system-wide aggregate (average across all GPUs)
    all_metrics = defaultdict(list)
    for typeId in data['gpus']:
        for metricId in NVLINK_METRICS:
            if metricId in data['gpu_metrics'][typeId]:
                all_metrics[metricId].extend(data['gpu_metrics'][typeId][metricId][:, 1])
    
    stats['aggregate']['metrics'] = {}
    for metricId, values in all_metrics.items():
        if values:
            stats['aggregate']['metrics'][NVLINK_METRICS[metricId]] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
            }
    
    # Calculate time-series data (averaged across GPUs at each timestamp)
    timestamps = data['timestamps']
    for cat_name, metric_ids in NVLINK_AGGREGATE.items():
        ts_values = []
        for ts in timestamps:
            ts_sum = []
            for typeId in data['gpus']:
                for mid in metric_ids:
                    if typeId in data['timestamp_data'][ts] and mid in data['timestamp_data'][ts][typeId]:
                        ts_sum.append(data['timestamp_data'][ts][typeId][mid])
            ts_values.append(np.mean(ts_sum) if ts_sum else 0)
        stats['time_series'][cat_name] = np.array(ts_values)
    
    stats['time_series']['timestamps'] = timestamps
    
    return stats


def print_stats(stats):
    """Print NVLink utilization statistics"""
    print("\n" + "=" * 80)
    print("NVLink Bandwidth Utilization Analysis")
    print("=" * 80)
    
    # Print aggregate stats
    print("\n--- System-wide Aggregate Statistics (all GPUs) ---")
    print(f"{'Metric':<45} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}")
    print("-" * 80)
    
    if 'metrics' in stats['aggregate']:
        for metric_name, mstats in sorted(stats['aggregate']['metrics'].items()):
            print(f"{metric_name:<45} {mstats['mean']:>7.1f}% {mstats['p50']:>7.1f}% "
                  f"{mstats['p95']:>7.1f}% {mstats['max']:>7.1f}%")
    
    # Print per-GPU summary
    print("\n--- Per-GPU Summary ---")
    for gpu_name, gpu_stats in sorted(stats['per_gpu'].items()):
        print(f"\n{gpu_name}:")
        
        # Print aggregate categories
        for cat_name in ['Total RX', 'Total TX', 'Total User Data', 'Total Protocol']:
            if cat_name in gpu_stats:
                cstats = gpu_stats[cat_name]
                print(f"  {cat_name:<25}: mean={cstats['mean']:>5.1f}%, "
                      f"p95={cstats['p95']:>5.1f}%, max={cstats['max']:>5.1f}%")
    
    print("\n" + "=" * 80)


def find_active_region(values, threshold=0.5):
    """
    Find the start and end indices of the active region (non-zero utilization)
    
    Args:
        values: Array of utilization values
        threshold: Minimum value to consider as "active"
    
    Returns:
        tuple: (start_idx, end_idx) of active region
    """
    # Find first non-zero index
    active_mask = values > threshold
    if not np.any(active_mask):
        return 0, len(values)
    
    active_indices = np.where(active_mask)[0]
    start_idx = max(0, active_indices[0] - 10)  # Add small margin
    end_idx = min(len(values), active_indices[-1] + 10)
    
    return start_idx, end_idx


def plot_utilization(stats, output_file=None, title=None):
    """
    Generate NVLink bandwidth utilization visualization
    
    Creates a clean time-series plot showing utilization over time,
    focusing only on the active region (non-zero utilization period).
    """
    if not stats or 'time_series' not in stats:
        print("No data available for plotting")
        return
    
    # Import matplotlib
    plt = _import_matplotlib()
    
    # Set up the figure with a clean dark theme
    plt.style.use('dark_background')
    fig, ax = plt.figure(figsize=(14, 6)), plt.gca()
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Custom colors - vibrant palette
    colors = {
        'Total RX': '#58a6ff',      # Blue
        'Total TX': '#f97583',      # Red/Pink
        'Total User Data': '#7ee787',  # Green
    }
    
    timestamps = stats['time_series']['timestamps']
    
    # Combine all values to find active region
    all_values = np.zeros_like(timestamps, dtype=float)
    for cat_name in ['Total RX', 'Total TX', 'Total User Data']:
        if cat_name in stats['time_series']:
            all_values = np.maximum(all_values, stats['time_series'][cat_name])
    
    # Find active region (trim zeros at start and end)
    start_idx, end_idx = find_active_region(all_values, threshold=0.5)
    
    # Slice data to active region only
    timestamps_active = timestamps[start_idx:end_idx]
    time_sec = (timestamps_active - timestamps_active[0]) / 1e9  # Convert to seconds
    
    print(f"  Active region: {start_idx} to {end_idx} ({len(time_sec)} samples)")
    print(f"  Time range: {time_sec[0]:.2f}s to {time_sec[-1]:.2f}s ({time_sec[-1] - time_sec[0]:.2f}s total)")
    
    # Plot each category
    for cat_name in ['Total RX', 'Total TX', 'Total User Data']:
        if cat_name in stats['time_series']:
            values = stats['time_series'][cat_name][start_idx:end_idx]
            label = cat_name.replace('Total ', '')
            ax.plot(time_sec, values, label=label, color=colors[cat_name], 
                    linewidth=1.2, alpha=0.9)
    
    # Style the plot
    ax.set_xlabel('Time (seconds)', fontsize=12, color='#c9d1d9')
    ax.set_ylabel('NVLink Throughput (%)', fontsize=12, color='#c9d1d9')
    
    fig_title = title or 'NVLink Bandwidth Utilization Over Time'
    ax.set_title(fig_title, fontsize=14, fontweight='bold', color='#f0f6fc', pad=15)
    
    # Legend
    legend = ax.legend(loc='upper right', framealpha=0.9, facecolor='#161b22', 
                       edgecolor='#30363d', fontsize=10)
    for text in legend.get_texts():
        text.set_color('#c9d1d9')
    
    # Grid
    ax.grid(True, alpha=0.2, color='#30363d', linestyle='-')
    ax.set_axisbelow(True)
    
    # Set y-axis limit
    max_val = max(all_values[start_idx:end_idx]) if len(all_values) > 0 else 100
    ax.set_ylim(0, min(100, max_val * 1.1))
    ax.set_xlim(time_sec[0], time_sec[-1])
    
    # Style axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.tick_params(colors='#8b949e')
    
    # Add statistics annotation
    stats_text = []
    for cat_name in ['Total RX', 'Total TX', 'Total User Data']:
        if cat_name in stats['time_series']:
            values = stats['time_series'][cat_name][start_idx:end_idx]
            values_nonzero = values[values > 0]
            if len(values_nonzero) > 0:
                label = cat_name.replace('Total ', '')
                mean_val = np.mean(values_nonzero)
                p95_val = np.percentile(values_nonzero, 95)
                stats_text.append(f"{label}: mean={mean_val:.1f}%, p95={p95_val:.1f}%")
    
    if stats_text:
        text_box = ax.text(0.02, 0.98, '\n'.join(stats_text), 
                          transform=ax.transAxes, fontsize=9,
                          verticalalignment='top', fontfamily='monospace',
                          color='#8b949e',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', 
                                   edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def analyze_nvlink_bandwidth(input_file, output_plot=None):
    """
    Main analysis function
    
    Args:
        input_file: Path to sqlite or nsys-rep file
        output_plot: Optional path for output plot
    
    Returns:
        dict: Analysis statistics
    """
    # Handle nsys-rep files
    if input_file.endswith('.nsys-rep'):
        sqlite_file = export_nsys_to_sqlite(input_file)
        if not sqlite_file:
            return None
        cleanup_sqlite = True
    else:
        sqlite_file = input_file
        cleanup_sqlite = False
    
    # Load metrics
    data = load_nvlink_metrics(sqlite_file)
    if not data:
        return None
    
    # Calculate statistics
    stats = calculate_utilization_stats(data)
    
    # Print results
    print_stats(stats)
    
    # Generate plot
    title = f'NVLink Bandwidth Utilization: {os.path.basename(input_file)}'
    plot_utilization(stats, output_plot, title)
    
    return stats


def metric_cal(directory, output_plot=None):
    """
    CCL-Bench compatible interface
    
    Args:
        directory: Directory containing sqlite/nsys-rep files, or single file path
        output_plot: Optional path for output plot
    
    Returns:
        dict: NVLink utilization statistics
    """
    # Find input file
    input_file = None
    
    if os.path.isfile(directory):
        input_file = directory
    elif os.path.isdir(directory):
        # Look for sqlite files first
        for fname in os.listdir(directory):
            if fname.endswith('.sqlite'):
                input_file = os.path.join(directory, fname)
                break
        # Then nsys-rep
        if not input_file:
            for fname in os.listdir(directory):
                if fname.endswith('.nsys-rep'):
                    input_file = os.path.join(directory, fname)
                    break
    
    if not input_file:
        print(f"No sqlite or nsys-rep files found in {directory}")
        return {}
    
    stats = analyze_nvlink_bandwidth(input_file, output_plot)
    return stats or {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze NVLink bandwidth utilization from nsys profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze sqlite file
    python nvlink_bandwidth_analyzer.py $PSCRATCH/datas/llama_bandwidth_fixed.sqlite
    
    # Analyze nsys-rep file
    python nvlink_bandwidth_analyzer.py profile.nsys-rep
    
    # Specify output plot
    python nvlink_bandwidth_analyzer.py data.sqlite --output nvlink_util.png
    
    # Analyze directory
    python nvlink_bandwidth_analyzer.py $PSCRATCH/datas/ --output results.png
"""
    )
    
    parser.add_argument('input', help='SQLite file, nsys-rep file, or directory containing them')
    parser.add_argument('--output', '-o', help='Output plot filename (PNG)', default=None)
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified
    output_file = args.output
    if not output_file:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_file = f'{base_name}_nvlink_utilization.png'
    
    stats = metric_cal(args.input, output_file)
    
    if stats:
        print(f"\n=== Summary ===")
        if 'aggregate' in stats and 'metrics' in stats['aggregate']:
            for cat in ['Total RX', 'Total TX']:
                for gpu_name, gpu_stats in stats['per_gpu'].items():
                    if cat in gpu_stats:
                        print(f"{gpu_name} {cat}: mean={gpu_stats[cat]['mean']:.1f}%, p95={gpu_stats[cat]['p95']:.1f}%")

