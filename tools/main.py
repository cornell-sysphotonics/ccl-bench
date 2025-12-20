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
    elif metric_name == "comm_kernel_breakdown_tpu":
        from comm_kernel_breakdown_tpu.comm_kernel_breakdown_tpu import comm_kernel_breakdown_tpu
        metric_cal_func = comm_kernel_breakdown_tpu
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    
    metric = metric_cal_func(trace_directory)
    print(metric)

