import json
import os
from typing import List, Dict

def _get_kineto_files(directory: str) -> List[str]:
    """
    Find all kineto_trace_*.json or kineto_rank*.json files in the trace directory.
    """
    files = []
    for fname in os.listdir(directory):
        if (fname.startswith("kineto_trace_") or fname.startswith("kineto_rank")) and fname.endswith(".json"):
            files.append(os.path.join(directory, fname))
    if not files:
        raise FileNotFoundError(f"No kineto trace files found in {directory}")
    return sorted(files)

def traffic_window_cal(directory: str) -> Dict[str, float]:
    """
    Calculates the 'Traffic Window' (T_comm): time spent in communication per iteration.
    Uses temporal heuristics to distinguish TP vs DP AllReduces.
    """
    kineto_files = _get_kineto_files(directory)
    
    # 1. Identify iteration boundaries from Rank 0
    with open(kineto_files[0], "r") as f:
        rank_0_data = json.load(f)
    
    # Standard marker for the backward/redistribute phase in Llama 3 / TorchTitan
    iteration_marker = "RedistributeBackward"
    iter_boundaries = []
    for event in rank_0_data.get("traceEvents", []):
        if event.get("name") == iteration_marker and event.get("cat") == "cpu_op":
            ts = event.get("ts")
            dur = event.get("dur", 0)
            # Boundary of the end-of-step synchronization phase
            iter_boundaries.append((ts, ts + dur))
    
    num_iters = max(len(iter_boundaries), 1)
    num_gpus = len(kineto_files)

    comm_stats = {
        "tp_window_sec": 0.0,
        "dp_window_sec": 0.0,
        "pp_window_sec": 0.0,
        "other_comm_window_sec": 0.0,
        "total_comm_window_sec": 0.0
    }

    # 2. Extract and categorize GPU communication kernels
    for path in kineto_files:
        with open(path, "r") as f:
            data = json.load(f)
            
        for event in data.get("traceEvents", []):
            name = event.get("name", "").lower()
            # Only count GPU Kernels (actual hardware time), not CPU launch ops
            if "nccl" in name and event.get("cat") == "kernel":
                ts = event.get("ts")
                dur_sec = event.get("dur", 0) / 1e6
                
                # Default categorization
                p_type = "other_comm_window_sec"
                
                if "allreduce" in name:
                    # TEMPORAL HEURISTIC:
                    # DP AllReduce happens specifically during the Redistribute/Sync phase at step end.
                    # TP AllReduce happens frequently between those phases (during FWD/BWD layers).
                    is_in_sync_phase = any(start <= ts <= end for start, end in iter_boundaries)
                    p_type = "dp_window_sec" if is_in_sync_phase else "tp_window_sec"
                
                elif "allgather" in name or "reduce_scatter" in name or "redistribute" in name:
                    # FSDP style traffic
                    p_type = "dp_window_sec"
                
                elif "send" in name or "recv" in name:
                    # Pipeline Parallel point-to-point traffic
                    p_type = "pp_window_sec"

                # Aggregate stats (no double counting)
                comm_stats[p_type] += dur_sec
                comm_stats["total_comm_window_sec"] += dur_sec

    # 3. Average per iteration and per GPU
    final_metrics = {
        k: round(v / (num_iters * num_gpus), 6) for k, v in comm_stats.items()
    }

    return final_metrics