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


if __name__ == "__main__":
    trace_directory = None
    metric_name = None
    metric_cal_func = None

    parser = argparse.ArgumentParser(description="Process trace directory and metric name.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace directory")
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
    
    
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    
    print(metric)

