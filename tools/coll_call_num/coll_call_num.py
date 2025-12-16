import json
from pathlib import Path

from tools.common.detect_profile_mode import available_profile_modes, list_torch_trace_dirs


def _load_trace(trace_file: Path) -> dict | str:
    try:
        with trace_file.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"Trace file not found: {trace_file}"
    except json.JSONDecodeError:
        return f"Error decoding JSON in file: {trace_file}"
    else:
        if isinstance(data, dict):
            return data
        return f"Unexpected trace format in file: {trace_file}"


def metric_cal(directory: str, profile_mode: str = "auto") -> int | dict:
    """Calculate the number of NCCL communication calls from kineto traces.

    For non-torch (e.g., nsys) traces, returns an error payload instead of failing.
    """
    if profile_mode == "auto":
        modes = available_profile_modes(directory)
        profile_mode = "torch" if "torch" in modes else modes[0]

    if profile_mode != "torch":
        modes = available_profile_modes(directory)
        if "torch" in modes:
            profile_mode = "torch"
        else:
            return {"error": "coll_call_num is only supported for torch/kineto traces"}

    torch_dirs = list_torch_trace_dirs(directory)
    if not torch_dirs:
        return {"error": "No torch trace JSON files found (expected *trace.json)"}

    communication_calls = 0

    for torch_dir in torch_dirs:
        candidate_files = list(torch_dir.glob("*trace.json"))
        if not candidate_files:
            legacy = torch_dir / "kineto_trace_0.json"
            candidate_files = [legacy] if legacy.exists() else []

        for trace_file in candidate_files:
            trace_data = _load_trace(trace_file)
            if isinstance(trace_data, str):
                return {"error": trace_data}

            for event in trace_data.get("traceEvents", []):
                name = event.get("name", "")
                if "nccl" in name.lower():
                    communication_calls += 1

    return communication_calls
