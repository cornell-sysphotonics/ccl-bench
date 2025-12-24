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
    elif metric_name == "communication_ratio":
        from communication_ratio_group_9.communication_ratio_group_9  import compute_comm_ratio
        metric_cal_func = compute_comm_ratio
    elif metric_name == "total_kernel_time":
        from total_kernel_time_group_9.total_kernel_time_group_9  import compute_total_kernel_time
        metric_cal_func = compute_total_kernel_time
    elif metric_name == "total_communication_time":
        from total_communication_time_group_9.total_communication_time_group_9  import compute_total_comm_time
        metric_cal_func = compute_total_comm_time
    elif metric_name == "break_down_steps":
        from break_down_steps_group_9.break_down_steps_group_9  import compute_breakdown
        metric_cal_func = compute_breakdown
    elif metric_name == "communication_fraction":
        from communication_fraction_group_9.communication_fraction_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "moe_fraction":
        from moe_fraction_group_9.moe_fraction_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "dominant_kernel_concentration":
        from dominant_kernel_concentration_group_9.dominant_kernel_concentration_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "aggregate_gpu_utilization":
        from aggregate_gpu_utilization_group_9.aggregate_gpu_utilization_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "mean_sm_coverage":
        from mean_sm_coverage_group_9.mean_sm_coverage_group_9 import calculate_metric
        metric_cal_func = calculate_metric
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
        from load_imbalance_ratio_group_9.load_imbalance_ratio_group_9 import calculate_metric
        metric_cal_func = calculate_metric
    elif metric_name == "communication_overlap_ratio":
        from communication_overlap_ratio_group_9.communication_overlap_ratio_group_9 import calculate_metric
        metric_cal_func = calculate_metric
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
        return
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
    elif metric_name == "mfu":
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
    
    metric = metric_cal_func(trace_directory)
    print(metric)
