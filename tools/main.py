#!/usr/bin/env python3
"""
Main entry point for CCL-bench tools and metrics.

This follows the CCL-bench standard interface for metric calculation tools.
"""

import argparse

if __name__ == "__main__":
    trace_directory = None
    metric_name = None
    metric_cal_func = None

    parser = argparse.ArgumentParser(description="Process trace directory and metric name.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory (or CSV results directory)")
    parser.add_argument("--metric", type=str, required=True, help="Name of the metric to calculate")

    args = parser.parse_args()

    trace_directory = args.trace
    metric_name = args.metric
    
    if metric_name == "coll_call_num":
        from coll_call_num.coll_call_num import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "comm_kernel_breakdown_tpu":
        from comm_kernel_breakdown_tpu_group_4.comm_kernel_breakdown_tpu_group_4 import comm_kernel_breakdown_tpu
        metric_cal_func = comm_kernel_breakdown_tpu
    elif metric_name == "ttft":
        from ttft_group_4.ttft import ttft
        metric_cal_func = ttft
    elif metric_name == "tpot":
        from tpot_group_4.tpot import tpot
        metric_cal_func = tpot
    elif metric_name == "mfu":
        from mfu_group_4.mfu import mfu
        metric_cal_func = mfu
    elif metric_name == "estimated_bandwidth":
        from estimated_bandwidth_group_4.estimated_bandwidth import estimated_bandwidth
        metric_cal_func = estimated_bandwidth
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    
    metric = metric_cal_func(trace_directory)
    print(metric)
