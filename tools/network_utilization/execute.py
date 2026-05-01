import json
from pathlib import Path

from trace_utils import extract_metrics_from_trace

DTYPE_BYTES = {
    "Float": 4, "Float32": 4,
    "Double": 8, "Float64": 8,
    "Half": 2, "Float16": 2, "BFloat16": 2, "BF16": 2,
    "Int": 4, "Int32": 4,
    "Int64": 8, "Long": 8,
    "Int16": 2, "Short": 2,
    "Int8": 1, "Byte": 1,
}

TPU_HLO_CATEGORY = {
    "all-reduce": "allreduce",
    "all-gather": "allgather",
    "reduce-scatter": "reducescatter",
    "all-to-all": "alltoall",
}


def load_events(trace_path: str) -> list:
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


def _is_kernel(event: dict) -> bool:
    return event.get("ph") == "X" and event.get("cat") == "kernel"


def _event_bytes(event: dict) -> int:
    args = event.get("args", {})
    in_nelems = args.get("In msg nelems")
    dtype = args.get("dtype")
    if in_nelems is None or dtype is None:
        return 0
    elem_bytes = DTYPE_BYTES.get(dtype, 0)
    return int(in_nelems) * elem_bytes

def _collective_check_helper(event: dict, kernel_names: list[str], collective_names: list[str]):
    name: str = event.get("name", "")
    collective_name = event.get("args", {}).get("Collective name", "")

    if any(name.startswith(prefix) for prefix in kernel_names):
        return True
    
    return collective_name in collective_names


def _is_allreduce(event: dict) -> bool:
    if not _is_kernel(event):
        return False

    return _collective_check_helper(event, kernel_names=["ncclDevKernel_AllReduce", "ncclKernel_AllReduce"], collective_names=["all_reduce", "allreduce"])

def _is_allgather(event: dict) -> bool:
    if not _is_kernel(event):
        return False
    return _collective_check_helper(event, kernel_names=["ncclDevKernel_AllGather", "ncclKernel_AllGather"], collective_names=["all_gather", "allgather"])


def _is_reducescatter(event: dict) -> bool:
    if not _is_kernel(event):
        return False
    return _collective_check_helper(event, kernel_names=["ncclDevKernel_ReduceScatter", "ncclKernel_ReduceScatter"], collective_names=["reduce_scatter", "reducescatter"])


def _is_alltoall(event: dict) -> bool:
    if not _is_kernel(event):
        return False
    name = event.get("name", "")
    collective_name = event.get("args", {}).get("Collective name", "")
    is_sendrecv = name.startswith("ncclDevKernel_SendRecv") or name.startswith("ncclKernel_SendRecv")
    is_alltoall_name = collective_name in ("all_to_allv", "all_to_all", "alltoall")
    return is_sendrecv and is_alltoall_name


def get_collective_bytes(trace_path: str) -> dict:
    """Return total bytes per collective type for a single rank trace file."""
    totals = {"allreduce": 0, "allgather": 0, "reducescatter": 0, "alltoall": 0}
    predicates = {
        "allreduce": _is_allreduce,
        "allgather": _is_allgather,
        "reducescatter": _is_reducescatter,
        "alltoall": _is_alltoall,
    }
    for event in load_events(trace_path):
        for name, predicate in predicates.items():
            if predicate(event):
                totals[name] += _event_bytes(event)
                break
    return totals


def get_collective_bytes_tpu(trace_path: str) -> dict:
    """Return total bytes per collective type for a single TPU trace file."""
    totals = {"allreduce": 0, "allgather": 0, "reducescatter": 0, "alltoall": 0}
    for event in load_events(trace_path):
        if event.get("ph") != "X":
            continue
        hlo_category = event.get("args", {}).get("hlo_category", "")
        collective = TPU_HLO_CATEGORY.get(hlo_category)
        if collective is None:
            continue
        bytes_accessed = event.get("args", {}).get("bytes_accessed")
        if bytes_accessed is not None:
            totals[collective] += int(bytes_accessed)
    return totals


def _find_rank0_trace(directory: str) -> Path | None:
    d = Path(directory)
    candidates = sorted(d.glob("rank0_trace.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(d.glob("rank*_trace.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(d.glob("kineto_trace_*.json"))
    if candidates:
        return candidates[0]
    return None


def metric_cal(directory: str) -> dict:
    """
    Return total bytes per collective for the first rank in the trace directory.

    Returns a dict with keys: allreduce, allgather, reducescatter, alltoall.
    Values are total bytes (int). Returns all zeros if no trace is found.
    """
    d = Path(directory)
    if "tpu" in d.name.lower():
        trace_files = sorted(d.glob("*.json"))
        if not trace_files:
            print(f"No TPU trace found in {directory}")
            return {"allreduce": 0, "allgather": 0, "reducescatter": 0, "alltoall": 0, "total_time_s": 0.0}
        trace_path = str(trace_files[0])
        result = get_collective_bytes_tpu(trace_path)
        result["total_time_s"] = extract_metrics_from_trace(trace_path).get("wall_s", 0.0)
        return result

    trace_path = _find_rank0_trace(directory)
    if trace_path is None:
        print(f"No GPU trace found in {directory}")
        return {"allreduce": 0, "allgather": 0, "reducescatter": 0, "alltoall": 0, "total_time_s": 0.0}
    trace_path = str(trace_path)
    result = get_collective_bytes(trace_path)
    result["total_time_s"] = extract_metrics_from_trace(trace_path).get("wall_s", 0.0)
    return result
