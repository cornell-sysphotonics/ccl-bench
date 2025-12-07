import json
from typing import List
import os


def metric_cal(directory: str) -> int:
    """
    Calculate the number of communication calls from PyTorch ET trace files in a directory.

    Args:
        directory (str): Path to the directory containing PyTorch ET trace JSON files.

    Returns:
        int: Total number of communication calls.
    """

    #TODO: perform trace metadata check. For example, check if the trace is from NVIDIA GPU and uses NCCL for communication.
    communication_calls = 0
    trace_file = os.path.join(directory, "kineto_trace_0.json")
    comm_name = ["ncclDevKernel_AllReduce", "ncclDevKernel_ReduceScatter", "ncclDevKernel_AllGather", "ncclDevKernel_Broadcast", "ncclDevKernel_Reduce", "ncclDevKernel_SendRecv"]

    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
            # Assuming communication calls are identified by a specific event type
            for event in trace_data.get("traceEvents", []):
                if event.get("cat") == "kernel" and any(event.get("name", "").startswith(name) for name in comm_name):  
                    communication_calls += 1
    except FileNotFoundError:
        print(f"File not found: {trace_file}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {trace_file}")

    return communication_calls

