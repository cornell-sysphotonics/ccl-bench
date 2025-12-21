import argparse
import json

# Available metrics and their modules
METRIC_MODULES = {
    "coll_call_num": "coll_call_num.coll_call_num",
    # NVLink throughput metrics
    "max_throughput": "nvlink_usage.analyze_nvlink_throughput",
    "avg_throughput": "nvlink_usage.analyze_nvlink_throughput",
    "total_communication": "nvlink_usage.analyze_nvlink_throughput",
    "nvlink_all": "nvlink_usage.analyze_nvlink_throughput",
    # NVLink kernel correlation metrics
    "peak_kernels": "nvlink_kernel_correlation.correlate_nsys_nvlink",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trace directory and calculate metrics.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory")
    parser.add_argument("--metric", type=str, required=True, 
                        choices=list(METRIC_MODULES.keys()),
                        help="Name of the metric to calculate")
    parser.add_argument("--link", type=int, default=None,
                        help="NVLink link ID filter (for nvlink metrics)")
    parser.add_argument("--direction", type=str, choices=["tx", "rx"], default=None,
                        help="NVLink direction filter (for nvlink metrics)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID filter for kernel metrics (default: 0)")
    parser.add_argument("--top", type=int, default=1,
                        help="Number of peak intervals for peak_kernels (default: 1)")
    parser.add_argument("--nccl-only", action="store_true",
                        help="Only show NCCL kernels for peak_kernels")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()
    
    metric_name = args.metric
    
    if metric_name == "coll_call_num":
        from coll_call_num.coll_call_num import metric_cal
        result = metric_cal(args.trace)
    elif metric_name in ["max_throughput", "avg_throughput", "total_communication", "nvlink_all"]:
        from nvlink_usage.analyze_nvlink_throughput import metric_cal
        # Map nvlink_all to "all"
        nvlink_metric = "all" if metric_name == "nvlink_all" else metric_name
        result = metric_cal(args.trace, metric_name=nvlink_metric, 
                           link=args.link, direction=args.direction)
    elif metric_name == "peak_kernels":
        from nvlink_kernel_correlation.correlate_nsys_nvlink import metric_cal
        result = metric_cal(args.trace, metric_name=metric_name,
                           gpu=args.gpu, link=args.link, direction=args.direction,
                           top=args.top, nccl_only=getattr(args, 'nccl_only', False))
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)

