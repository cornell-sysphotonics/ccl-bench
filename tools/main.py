import argparse


if __name__ == "__main__":
    trace_directory = None
    metric_name = None
    metric_cal_func = None

    parser = argparse.ArgumentParser(description="Process trace directory and metric name.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory")
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
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    
    metric = metric_cal_func(trace_directory)
    print(metric)

