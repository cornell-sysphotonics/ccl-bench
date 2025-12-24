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