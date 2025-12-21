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
    elif metric_name == "throughput_tokens_sec":
        from throughput_tokens_sec.throughput_tokens_sec import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "iteration_wall_clock":
        from iteration_wall_clock.iteration_wall_clock import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "comm_comp_overlap":
        from comm_comp_overlap.comm_comp_overlap import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "comm_overhead":
        from comm_overhead.comm_overhead import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "ttft":
        from ttft.ttft import metric_cal
        metric_cal_func = metric_cal
    elif metric_name == "tpot":
        from tpot.tpot import metric_cal
        metric_cal_func = metric_cal
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    
    metric = metric_cal_func(trace_directory)
    print(metric)
