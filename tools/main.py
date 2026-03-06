#!/usr/bin/env python3
"""
Main entry point for CCL-bench tools and metrics.

This follows the CCL-bench standard interface for metric calculation tools.
"""

import argparse
import os
import glob
import importlib.util
import sys

def find_trace_file(trace_directory: str) -> str:
    """
    Tries to locate the proper file in the given directory, returns an 
    error if it cannot find the proper trace.json file 
    Note: this function looks ugly since we require a very specific 
    type of trace to make all our functions work 
    """
    if not os.path.exists(trace_directory):
        raise FileNotFoundError(f"Trace directory does not exist: {trace_directory}")
    
    # If it's a file (not a directory), return it directly
    if os.path.isfile(trace_directory):
        return trace_directory
    
    if not os.path.isdir(trace_directory):
        raise ValueError(f"Path is not a directory: {trace_directory}")
    
    # Helper function that checks if the path given is a file path 
    def is_file(path):
        return os.path.isfile(path)
    
    kineto_path = os.path.join(trace_directory, "kineto_trace_0.json")
    if os.path.exists(kineto_path) and is_file(kineto_path):
        return kineto_path
    trace_files = glob.glob(os.path.join(trace_directory, "**", "*.trace.json"), recursive=True)
    if not trace_files:
        trace_files = glob.glob(os.path.join(trace_directory, "*.trace.json"))
    trace_files = [f for f in trace_files if is_file(f)]
    if trace_files:
        return trace_files[0]
    
    gz_files = glob.glob(os.path.join(trace_directory, "**", "*.json.gz"), recursive=True)
    if not gz_files:
        gz_files = glob.glob(os.path.join(trace_directory, "*.json.gz"))
    # Filter to only actual files
    gz_files = [f for f in gz_files if is_file(f)]
    if gz_files:
        return gz_files[0]
    
    json_files = glob.glob(os.path.join(trace_directory, "**", "*.json"), recursive=True)
    if not json_files:
        json_files = glob.glob(os.path.join(trace_directory, "*.json"))
    json_files = [f for f in json_files if is_file(f) and not f.endswith("kineto_trace_0.json")]
    if json_files:
        return json_files[0]
    
    try:
        dir_contents = os.listdir(trace_directory)
        raise FileNotFoundError(
            f"Could not find trace JSON file in directory: {trace_directory}\n"
            f"Directory contents: {dir_contents}"
        )
    except PermissionError:
        raise FileNotFoundError(
            f"Could not find trace JSON file in directory: {trace_directory}\n"
            f"(Permission denied when trying to list directory contents)"
        )


def load_metric_module(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import statistics
import math
import importlib.util


if __name__ == "__main__":
    trace_directory = None
    metric_name = None
    metric_cal_func = None

    parser = argparse.ArgumentParser(description="Process trace directory and metric name.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory (or CSV results directory)")
    parser.add_argument("--metric", type=str, required=True, help="Name of the metric to calculate")
    parser.add_argument("--n_chips", type=int, default=None, help="Number of chips (required for bandwidth and hockney metrics)")
    parser.add_argument("--model_params", type=float, default=None, help="Model parameters for throughput estimation (optional)")

    args = parser.parse_args()

    trace_directory = args.trace
    metric_name = args.metric
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find trace file 
    try:
        trace_json_path = find_trace_file(trace_directory)
    except Exception as e:
        print(f"Error finding trace file in {trace_directory}: {e}", file=sys.stderr)
        raise
    
    if metric_name == "wall_time_s":
        wall_time_module = load_metric_module(os.path.join(tools_dir, "wall_time-group-21", "wall_time.py"), "wall_time")
        metric = wall_time_module.compute_metric(trace_json_path)
    
    elif metric_name == "total_compute_time_s":
        compute_time_module = load_metric_module(os.path.join(tools_dir, "compute_time-group-21", "compute_time.py"), "compute_time")
        metric = compute_time_module.compute_metric(trace_json_path)
    
    # Communication time metrics
    elif metric_name == "total_comm_time_s":
        comm_time_module = load_metric_module(os.path.join(tools_dir, "comm_time-group-21", "comm_time.py"), "comm_time")
        metric = comm_time_module.compute_metric(trace_json_path, metric_type="total")
    
    elif metric_name == "avg_comm_kernel_time_s":
        comm_time_module = load_metric_module(os.path.join(tools_dir, "comm_time-group-21", "comm_time.py"), "comm_time")
        metric = comm_time_module.compute_metric(trace_json_path, metric_type="avg_kernel")
    
    elif metric_name == "allreduce_comm_time_s":
        comm_time_module = load_metric_module(os.path.join(tools_dir, "comm_time-group-21", "comm_time.py"), "comm_time")
        metric = comm_time_module.compute_metric(trace_json_path, metric_type="allreduce")
    
    # Utilization metrics
    elif metric_name == "compute_utilization_proxy":
        utilization_module = load_metric_module(os.path.join(tools_dir, "utilization-group-21", "utilization.py"), "utilization")
        metric = utilization_module.compute_metric(trace_json_path, metric_type="compute_util")
    
    elif metric_name == "communication_fraction":
        utilization_module = load_metric_module(os.path.join(tools_dir, "utilization-group-21", "utilization.py"), "utilization")
        metric = utilization_module.compute_metric(trace_json_path, metric_type="comm_fraction")
    
    # Bandwidth metrics (require n_chips)
    elif metric_name == "num_comm_kernels":
        if args.n_chips is None:
            raise ValueError(f"Metric '{metric_name}' requires --n_chips parameter")
        bandwidth_module = load_metric_module(os.path.join(tools_dir, "bandwidth-group-21", "bandwidth.py"), "bandwidth")
        metric = bandwidth_module.compute_metric(trace_json_path, args.n_chips, metric_type="num_comm_kernels")
    
    
    elif metric_name == "avg_comm_bandwidth_GBps":
        if args.n_chips is None:
            raise ValueError(f"Metric '{metric_name}' requires --n_chips parameter")
        bandwidth_module = load_metric_module(os.path.join(tools_dir, "bandwidth-group-21", "bandwidth.py"), "bandwidth")
        metric = bandwidth_module.compute_metric(trace_json_path, args.n_chips, metric_type="avg_bandwidth")
    
    # Hockney metrics (require n_chips)
    elif metric_name == "hockney_alpha_s":
        if args.n_chips is None:
            raise ValueError(f"Metric '{metric_name}' requires --n_chips parameter")
        hockney_module = load_metric_module(os.path.join(tools_dir, "hockney-group-21", "hockney.py"), "hockney")
        metric = hockney_module.compute_metric(trace_json_path, args.n_chips, metric_type="alpha")
    
    elif metric_name == "hockney_beta_s_per_byte":
        if args.n_chips is None:
            raise ValueError(f"Metric '{metric_name}' requires --n_chips parameter")
        hockney_module = load_metric_module(os.path.join(tools_dir, "hockney-group-21", "hockney.py"), "hockney")
        metric = hockney_module.compute_metric(trace_json_path, args.n_chips, metric_type="beta")
    
    elif metric_name == "hockney_inverse_beta_Bps":
        if args.n_chips is None:
            raise ValueError(f"Metric '{metric_name}' requires --n_chips parameter")
        hockney_module = load_metric_module(os.path.join(tools_dir, "hockney-group-21", "hockney.py"), "hockney")
        metric = hockney_module.compute_metric(trace_json_path, args.n_chips, metric_type="inverse_beta")
    
    
    # FLOPs metrics
    elif metric_name == "achieved_flops_from_trace_json":
        flops_module = load_metric_module(os.path.join(tools_dir, "flops-group-21", "flops.py"), "flops")
        metric = flops_module.compute_metric(trace_json_path, metric_type="achieved_flops")
    
    elif metric_name == "total_model_flops_from_args":
        flops_module = load_metric_module(os.path.join(tools_dir, "flops-group-21", "flops.py"), "flops")
        metric = flops_module.compute_metric(trace_json_path, metric_type="total_model_flops")
    
    # Throughput metrics
    elif metric_name == "throughput":
        throughput_module = load_metric_module(os.path.join(tools_dir, "throughput-group-21", "throughput.py"), "throughput")
        model_params = args.model_params if args.model_params is not None else 7e9
        metric = throughput_module.compute_metric(trace_json_path, model_params=model_params)
    
    elif metric_name == "estimated_throughput_tokens_per_s":
        throughput_module = load_metric_module(os.path.join(tools_dir, "throughput-group-21", "throughput.py"), "throughput")
        model_params = args.model_params if args.model_params is not None else 7e9
        metric = throughput_module.compute_metric(trace_json_path, metric_type="estimated_throughput_tokens_per_s", model_params=model_params)
    
    elif metric_name == "flops_per_token_used":
        throughput_module = load_metric_module(os.path.join(tools_dir, "throughput-group-21", "throughput.py"), "throughput")
        model_params = args.model_params if args.model_params is not None else 7e9
        metric = throughput_module.compute_metric(trace_json_path, metric_type="flops_per_token_used", model_params=model_params)
    
    elif metric_name == "estimated_total_tokens":
        throughput_module = load_metric_module(os.path.join(tools_dir, "throughput-group-21", "throughput.py"), "throughput")
        model_params = args.model_params if args.model_params is not None else 7e9
        metric = throughput_module.compute_metric(trace_json_path, metric_type="estimated_total_tokens", model_params=model_params)
    
    
    elif metric_name == "coll_call_num":
        from coll_call_num.coll_call_num import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "mfu":
        from mfu.mfu import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "avg_step_time":
        from avg_step_time.avg_step_time import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "communication_ratio":
        from communication_ratio.communication_ratio import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "total_trace_time":
        from total_trace_time.total_trace_time import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "total_kernel_time":
        from total_kernel_time.total_kernel_time import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "total_communication_time":
        from total_communication_time.total_communication_time import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "break_down_steps":
        from break_down_steps_group_9.break_down_steps_group_9  import compute_breakdown
        metric_cal_func = compute_breakdown
    elif metric_name == "communication_fraction":
        from communication_fraction.communication_fraction import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "moe_fraction":
        from moe_fraction.moe_fraction import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "dominant_kernel_concentration":
        from dominant_kernel_concentration.dominant_kernel_concentration import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "aggregate_gpu_utilization":
        from aggregate_gpu_utilization.aggregate_gpu_utilization import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "mean_sm_coverage":
        from mean_sm_coverage.mean_sm_coverage import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "memory_transfer_overhead":
        from memory_transfer_overhead_group_9.memory_transfer_overhead_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "average_memory_bandwidth":
        from average_memory_bandwidth_group_9.average_memory_bandwidth_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "compute_bound_fraction":
        from compute_bound_fraction_group_9.compute_bound_fraction_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "memory_bound_fraction":
        from memory_bound_fraction_group_9.memory_bound_fraction_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "load_imbalance_ratio":
        from load_imbalance_ratio.load_imbalance_ratio import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "communication_overlap_ratio":
        from communication_overlap_ratio.communication_overlap_ratio import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "ttft_group_6":
        from ttft_group_6.ttft_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "tpot_group_6":
        from tpot_group_6.tpot_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "bandwidth_utilization_allgather_group_6":
        from bandwidth_utilization_allgather_group_6.bandwidth_utilization_allgather_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "bandwidth_utilization_allreduce_group_6":
        from bandwidth_utilization_allreduce_group_6.bandwidth_utilization_allreduce_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "bandwidth_utilization_alltoall_group_6":
        from bandwidth_utilization_alltoall_group_6.bandwidth_utilization_alltoall_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "bandwidth_utilization_peertopeer_group_6":
        from bandwidth_utilization_peertopeer_group_6.bandwidth_utilization_peertopeer_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "kernel_compute_time_group_6":
        from kernel_compute_time_group_6.kernel_compute_time_group_6 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "throughput_group_6":
        from throughput_group_6.throughput_group_6 import metric_cal
    elif metric_name == "straggler_metrics":
        # group 5
        from straggler.straggler_metrics import metric_cal
        delay, slowdown = metric_cal(trace_directory)
        print("Straggler Delay: ", delay)
        print("Straggler Slowdown: ", slowdown)
        sys.exit(0)
    elif metric_name == "comm_kernel_breakdown_tpu":
        from comm_kernel_breakdown_tpu_group_4.comm_kernel_breakdown_tpu_group_4 import comm_kernel_breakdown_tpu
        metric_cal_func = comm_kernel_breakdown_tpu
    elif metric_name == "ttft":
        from ttft_group_4.ttft import ttft
        metric_cal_func = ttft
    elif metric_name == "tpot":
        from tpot_group_4.tpot import tpot
        metric_cal_func = tpot
    elif metric_name == "estimated_bandwidth":
        from estimated_bandwidth_group_4.estimated_bandwidth import estimated_bandwidth
        metric_cal_func = estimated_bandwidth
    elif metric_name == "mfu_group_1":
        from mfu_group_1.mfu_group_1 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "traffic_window":
        from traffic_window_group_1.traffic_window_group_1 import traffic_window_cal
        metric_cal_func = traffic_window_cal
    elif metric_name == "communication_overhead":
        from communication_overhead_group_1.communication_overhead_group_1 import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "bandwidth_utilization":
        from bandwidth_utilization_group_1.bandwidth_utilization_group_1 import metric_cal
        metric_cal_func = metric_cal
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    print(metric)
