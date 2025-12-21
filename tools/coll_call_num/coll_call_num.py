import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trace_analyzer import TraceAnalyzer


def metric_cal(directory: str) -> int:
    """
    Calculate the number of communication calls from PyTorch ET trace files in a directory.

    Args:
        directory (str): Path to the directory containing PyTorch ET trace JSON files.

    Returns:
        int: Total number of communication calls.
    """

    #TODO: perform trace metadata check. For example, check if the trace is from NVIDIA GPU and uses NCCL for communication.
    csv_candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("cuda_gpu_trace") and name.endswith(".csv")
    ]
    json_candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("kineto_trace_") and name.endswith(".json")
    ]
    trace_files = sorted(csv_candidates or json_candidates)
    if not trace_files:
        raise FileNotFoundError(f"No cuda_gpu_trace*.csv or kineto_trace_*.json found under {directory}")

    communication_calls = 0
    comm_name = ["nccl", "ncclDevKernel_AllReduce", "ncclDevKernel_ReduceScatter", "ncclDevKernel_AllGather", "ncclDevKernel_Broadcast", "ncclDevKernel_Reduce", "ncclDevKernel_SendRecv"]

    for trace_file in trace_files:
        analyzer = TraceAnalyzer(trace_file)
        for event in analyzer.events:
            if not analyzer._is_kernel_event(event):
                continue
            name = event.get("name", "").lower()
            if any(name.startswith(prefix.lower()) for prefix in comm_name):
                communication_calls += 1

    return communication_calls
