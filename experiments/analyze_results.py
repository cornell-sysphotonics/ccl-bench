#!/usr/bin/env python3
"""
Analyze and compare results across all experiments
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path to allow imports if run directly
# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tools.trace_analyzer import TraceAnalyzer
    from tools.ttft.ttft import metric_cal as cal_ttft
    from tools.tpot.tpot import metric_cal as cal_tpot
    from tools.pipeline_bubble_ratio.pipeline_bubble_ratio import metric_cal as cal_bubble_ratio
except ImportError as e:
    print(f"Warning: Could not import tools: {e}")
    TraceAnalyzer = None


def collect_metrics(trace_collection_dir):
    """Collect metrics from all experiment traces."""

    results = []

    for trace_dir in Path(trace_collection_dir).iterdir():
        if not trace_dir.is_dir():
            continue

        # Read config
        config_path = trace_dir / "config.yaml"
        if not config_path.exists():
            # Fallback for older runs or different naming
            config_path = trace_dir / "workload_card.yaml"
            if not config_path.exists():
                continue

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Read timing stats
        timing_candidates = [
            trace_dir / "timing_stats_0.json",
            trace_dir / "timing_stats_rank0.json",
        ]
        timing_stats_path = next((p for p in timing_candidates if p.exists()), None)
        if not timing_stats_path:
            print(f"Skipping {trace_dir} (no timing_stats file)")
            continue

        with open(timing_stats_path, 'r') as f:
            timing_stats = json.load(f)

        # Extract key information based on config structure
        # Handle both new config format and potential legacy format
        if 'parallelism' in config:
            # New format
            parallelism = config['parallelism']
            model_name = config['model']['name']
            batch_size = config['data']['batch_size']
            seq_len = config['data']['seq_len']
            gpus = parallelism.get('tp', 1) * parallelism.get('pp', 1) * parallelism.get('dp_shard', 1)
            # Note: Total GPUs might not be explicitly in config, infer from parallelism
        else:
            # Legacy format (workload_card)
            parallelism = config['Model-executor']['model_plan_parallelization']
            model_name = config['workload']['model']['model_family']
            batch_size = config['workload']['data']['batch_size']
            seq_len = config['workload']['data']['seq_len']
            gpus = config['workload']['hardware']['xpu_spec']['total_count']
        
        workload_card_path = trace_dir / "workload_card.yaml"
        if workload_card_path.exists():
            with open(workload_card_path, "r") as f:
                workload_card = yaml.safe_load(f)
            # Override with workload card values if present (source of truth for run-time)
            try:
                batch_size = workload_card["workload"]["data"]["batch_size"]
                seq_len = workload_card["workload"]["data"]["seq_len"]
            except Exception:
                pass

        
        # New metrics using tools
        ttft_ms = None
        tpot_ms = None
        bubble_ratio = None
        
        try:
            # Check if tools are available
            if 'cal_ttft' in globals():
                ttft_ms = cal_ttft(str(trace_dir))
            if 'cal_tpot' in globals():
                tpot_ms = cal_tpot(str(trace_dir))
            if 'cal_bubble_ratio' in globals():
                bubble_ratio = cal_bubble_ratio(str(trace_dir))
        except Exception as e:
            print(f"Error calculating TTFT/TPOT/Bubble for {trace_dir.name}: {e}")

        # Analyze traces for other metrics derived directly from traces
        comm_overhead = None
        sm_efficiency = None
        
        if TraceAnalyzer:
            kineto_trace_path = trace_dir / f"kineto_trace_0.json"
            if kineto_trace_path.exists():
                try:
                    analyzer = TraceAnalyzer(str(kineto_trace_path))
                    comm_overhead = analyzer.calculate_comm_overhead()
                    # bubble_ratio is already calculated via tool
                    sm_efficiency = analyzer.calculate_sm_efficiency()
                except Exception as e:
                    print(f"Error analyzing trace for {trace_dir.name}: {e}")

        result = {
            'experiment': trace_dir.name,
            'model': model_name,
            'tp': parallelism.get('tp', 1),
            'pp': parallelism.get('pp', 1),
            'dp': parallelism.get('dp_shard', 1),
            'ep': parallelism.get('ep', 1),
            'gpus': gpus,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'avg_iter_time': timing_stats.get('avg_iteration_time', 0),
            'min_iter_time': timing_stats.get('min_iteration_time', 0),
            'max_iter_time': timing_stats.get('max_iteration_time', 0),
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
            'comm_overhead_pct': comm_overhead,
            'bubble_ratio_pct': bubble_ratio,
            'sm_efficiency_pct': sm_efficiency,
        }

        if result['avg_iter_time'] and result['avg_iter_time'] > 0:
            # Derive iteration rate directly from measured timing. Token count per step is unknown here.
            result['steps_per_sec'] = 1.0 / result['avg_iter_time']
        else:
            result['steps_per_sec'] = None

        results.append(result)

    return pd.DataFrame(results)


def plot_scaling_analysis(df):
    """Generate scaling analysis plots."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df.dropna(subset=['steps_per_sec'])
        if model_df.empty:
            continue
        ax.plot(model_df['gpus'], model_df['steps_per_sec'],
                marker='o', label=model)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Steps per second')
    ax.set_title('Iteration Rate (derived from avg_iter_time)')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df.dropna(subset=['avg_iter_time'])
        if model_df.empty:
            continue
        ax.plot(model_df['gpus'], model_df['avg_iter_time'],
                marker='s', label=model)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Iteration Time (s)')
    ax.set_title('Iteration Time vs GPUs')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('experiments/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved scaling analysis plot to experiments/scaling_analysis.png")


def generate_summary_table(df):
    """Generate summary table for all experiments."""

    summary_cols = [
        'experiment', 'model', 'tp', 'pp', 'ep', 'gpus',
        'steps_per_sec', 'ttft_ms', 'tpot_ms',
        'comm_overhead_pct', 'bubble_ratio_pct'
    ]

    summary = df[summary_cols].copy()
    summary = summary.sort_values(['model', 'gpus'])

    # Save to CSV
    summary.to_csv('experiments/results_summary.csv', index=False)
    print("\nResults Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to experiments/results_summary.csv")

    return summary


def main():
    trace_collection_dir = "trace_collection"

    if not os.path.exists(trace_collection_dir):
        print(f"Error: {trace_collection_dir} not found")
        return

    print("Collecting metrics from experiments...")
    df = collect_metrics(trace_collection_dir)

    if df.empty:
        print("No experiment results found!")
        return

    print(f"\nFound {len(df)} experiments")

    # Generate summary table
    summary = generate_summary_table(df)

    # Generate plots
    print("\nGenerating scaling analysis plots...")
    plot_scaling_analysis(df)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
