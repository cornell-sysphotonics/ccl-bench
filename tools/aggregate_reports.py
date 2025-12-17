#!/usr/bin/env python3
# Python 3.6 compatible
"""
Aggregate analysis reports from multiple workloads.

This script:
1. Copies report.json files from workload folders to a consolidated folder
2. Transforms each report to extract only summary data (no per_rank_stats)
3. Aggregates all metrics into a unified format: metric_name -> parallelism_type -> value
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Any, Optional


def get_parallelism_type(folder_name):
    # type: (str) -> str
    """
    Extract parallelism type from folder name.
    
    Examples:
        deepseek-v2-lite-torchtitan-fsdp-perlmutter-16 -> fsdp
        deepseek-v2-lite-torchtitan-fsdp+tp-perlmutter-16 -> fsdp+tp
        deepseek-v2-lite-torchtitan-4d-perlmutter-16 -> 4d
    """
    # Split by '-torchtitan-' and '-perlmutter'
    parts = folder_name.split('-torchtitan-')
    if len(parts) < 2:
        return folder_name
    
    after_torchtitan = parts[1]
    # Remove the perlmutter-XX suffix
    parallelism = after_torchtitan.split('-perlmutter')[0]
    return parallelism


def extract_summary_data(metric_data):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """
    Extract summary data from a metric, excluding per_rank_stats.
    """
    if metric_data is None:
        return {}
    
    summary = {}
    for key, value in metric_data.items():
        # Skip per_rank_stats and other detailed breakdowns
        if key in ['per_rank_stats', 'top_kernels']:
            continue
        
        # For nested dicts like operation_breakdown, type_breakdown, layer_breakdown
        # Keep them but simplify if they contain per-rank data
        if isinstance(value, dict):
            # Check if it's a breakdown we want to keep
            if key in ['operation_breakdown', 'type_breakdown', 'layer_breakdown_ms', 
                       'per_operation_stats', 'device_info', 'duration_stats',
                       'kernel_time_stats', 'communication_time_stats', 
                       'slowest_rank', 'fastest_rank']:
                summary[key] = value
            else:
                # For other dicts, include if they don't have nested rank data
                summary[key] = value
        else:
            summary[key] = value
    
    return summary


def transform_report(report_data):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """
    Transform a report to extract just the summary data from each metric.
    """
    transformed = {
        'profile_dir': report_data.get('profile_dir', ''),
        'workload_card': report_data.get('workload_card', ''),
        'iterations': report_data.get('iterations', []),
        'metrics': {}
    }
    
    metrics = report_data.get('metrics', [])
    for metric in metrics:
        tool_name = metric.get('tool_name', 'unknown')
        success = metric.get('success', False)
        
        if success and metric.get('data'):
            transformed['metrics'][tool_name] = extract_summary_data(metric['data'])
        else:
            transformed['metrics'][tool_name] = {
                'success': False,
                'error': metric.get('error', 'Unknown error')
            }
    
    return transformed


def aggregate_reports(reports):
    # type: (Dict[str, Dict[str, Any]]) -> Dict[str, Any]
    """
    Aggregate all reports into a unified format.
    
    Format: metric_name -> parallelism_type -> values
    """
    aggregated = {}
    
    # Get all unique metric names
    all_metrics = set()
    for parallelism, report in reports.items():
        if 'metrics' in report:
            all_metrics.update(report['metrics'].keys())
    
    # Build aggregated structure
    for metric_name in sorted(all_metrics):
        aggregated[metric_name] = {}
        
        for parallelism, report in sorted(reports.items()):
            if 'metrics' in report and metric_name in report['metrics']:
                aggregated[metric_name][parallelism] = report['metrics'][metric_name]
    
    return aggregated


def find_workload_dirs(trace_collection_dir, prefix):
    # type: (str, str) -> List[str]
    """
    Find all workload directories matching the prefix.
    """
    workload_dirs = []
    
    if not os.path.isdir(trace_collection_dir):
        print("Error: {} is not a directory".format(trace_collection_dir))
        return workload_dirs
    
    for item in os.listdir(trace_collection_dir):
        item_path = os.path.join(trace_collection_dir, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            workload_dirs.append(item_path)
    
    return sorted(workload_dirs)


def main():
    # type: () -> int
    parser = argparse.ArgumentParser(
        description='Aggregate analysis reports from multiple workloads'
    )
    parser.add_argument(
        '--trace-collection-dir',
        type=str,
        required=True,
        help='Path to the trace_collection directory'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='deepseek',
        help='Prefix to filter workload directories (default: deepseek)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for consolidated reports (default: <trace_collection_dir>/aggregated_reports)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='aggregated_metrics.json',
        help='Name of the final aggregated JSON file (default: aggregated_metrics.json)'
    )
    
    args = parser.parse_args()
    
    trace_collection_dir = os.path.abspath(args.trace_collection_dir)
    output_dir = args.output_dir or os.path.join(trace_collection_dir, 'aggregated_reports')
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: {}".format(output_dir))
    
    # Find workload directories
    workload_dirs = find_workload_dirs(trace_collection_dir, args.prefix)
    print("Found {} workload directories matching prefix '{}'".format(
        len(workload_dirs), args.prefix
    ))
    
    if not workload_dirs:
        print("No workload directories found!")
        return 1
    
    # Step 1 & 2: Copy and transform reports
    copied_reports_dir = os.path.join(output_dir, 'individual_reports')
    if not os.path.exists(copied_reports_dir):
        os.makedirs(copied_reports_dir)
    
    transformed_reports = {}  # parallelism_type -> transformed_data
    
    for workload_dir in workload_dirs:
        workload_name = os.path.basename(workload_dir)
        report_path = os.path.join(workload_dir, 'analysis_output', 'report.json')
        
        if not os.path.exists(report_path):
            print("  SKIP: {} (no report.json found)".format(workload_name))
            continue
        
        # Copy report with unique name
        dest_name = "{}_report.json".format(workload_name)
        dest_path = os.path.join(copied_reports_dir, dest_name)
        shutil.copy2(report_path, dest_path)
        print("  Copied: {} -> {}".format(workload_name, dest_name))
        
        # Load and transform
        try:
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            parallelism_type = get_parallelism_type(workload_name)
            transformed = transform_report(report_data)
            transformed['workload_name'] = workload_name
            transformed['parallelism_type'] = parallelism_type
            
            transformed_reports[parallelism_type] = transformed
            
            # Save transformed report
            transformed_path = os.path.join(copied_reports_dir, 
                                           "{}_transformed.json".format(workload_name))
            with open(transformed_path, 'w') as f:
                json.dump(transformed, f, indent=2)
            
        except Exception as e:
            print("  ERROR processing {}: {}".format(workload_name, str(e)))
    
    if not transformed_reports:
        print("\nNo reports were successfully processed!")
        return 1
    
    print("\nSuccessfully processed {} reports".format(len(transformed_reports)))
    
    # Step 3: Aggregate all transformed reports
    print("\nAggregating metrics...")
    aggregated = aggregate_reports(transformed_reports)
    
    # Add metadata
    final_output = {
        'metadata': {
            'source_directory': trace_collection_dir,
            'prefix': args.prefix,
            'num_workloads': len(transformed_reports),
            'parallelism_types': sorted(transformed_reports.keys()),
            'metrics_available': sorted(aggregated.keys())
        },
        'aggregated_metrics': aggregated
    }
    
    # Save final aggregated file
    output_file = os.path.join(output_dir, args.output_file)
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print("\nAggregated metrics saved to: {}".format(output_file))
    print("\nSummary:")
    print("  - Parallelism types: {}".format(', '.join(sorted(transformed_reports.keys()))))
    print("  - Metrics aggregated: {}".format(', '.join(sorted(aggregated.keys()))))
    
    # Print a quick preview
    print("\n--- Quick Preview ---")
    for metric_name in sorted(aggregated.keys()):
        print("\n{}:".format(metric_name))
        for parallelism in sorted(aggregated[metric_name].keys()):
            data = aggregated[metric_name][parallelism]
            # Show first few key metrics
            preview_keys = list(data.keys())[:3]
            preview = {k: data[k] for k in preview_keys if not isinstance(data[k], dict)}
            print("  {}: {}".format(parallelism, preview))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

