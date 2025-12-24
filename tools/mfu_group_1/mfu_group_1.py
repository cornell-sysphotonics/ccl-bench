import json
import os
import re
from typing import List, Tuple, Dict

# Hardware Peak (A100)
A100_PEAK_TFLOPS_BF16 = 312.0
PEAK_FLOPS_PER_GPU = A100_PEAK_TFLOPS_BF16 * 1e12

def _parse_shell_script(script_path: str) -> Dict:
    """Extracts training constants from the Slurm/Bash submit script."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Submit script not found at {script_path}")
        
    with open(script_path, 'r') as f:
        content = f.read()

    # Regex patterns to find variable assignments or command line flags
    # This covers both 'VAR=value' and '--flag value' formats
    patterns = {
        "B": r"(?:GLOBAL_BATCH_SIZE=|--global-batch-size\s+)(\d+)",
        "S": r"(?:SEQ_LENGTH=|--seq-length\s+)(\d+)",
        "L": r"(?:NUM_LAYERS=|--num-layers\s+)(\d+)",
        "H": r"(?:HIDDEN_SIZE=|--hidden-size\s+)(\d+)",
        "world_size": r"(?:WORLD_SIZE=|--nproc_per_node\s+\d+\s+--nnodes\s+)(\d+)"
    }

    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            params[key] = int(match.group(1))
        else:
            # Fallbacks for Llama 3.1 8B defaults if not found in script
            defaults = {"B": 32, "S": 1024, "L": 32, "H": 4096, "world_size": 8}
            params[key] = defaults[key]

    return params

def _flops_per_iteration_global(params: Dict) -> float:
    """
    Theoretical FLOPs ≈ 2 * B * S * L * H^2
    Note: We use the values extracted from the .sh script.
    """
    return 2.0 * params['B'] * params['S'] * params['L'] * (params['H'] ** 2)

def _get_kineto_files(directory: str) -> List[str]:
    files = [os.path.join(directory, f) for f in os.listdir(directory) 
             if (f.startswith("kineto_trace_") or f.startswith("kineto_rank")) and f.endswith(".json")]
    if not files:
        raise FileNotFoundError(f"No traces found in {directory}")
    return sorted(files)

def _global_iteration_time_seconds(directory: str) -> Tuple[float, int, int, str]:
    kineto_files = _get_kineto_files(directory)
    rank_0_path = kineto_files[0]
    with open(rank_0_path, "r") as f:
        data = json.load(f)

    # Markers: Added 'fwdbwd' and generic 'nccl' launch points
    markers = ["RedistributeBackward", "fwdbwd", "Optimizer.step", "ncclDevKernel"]
    dynamic_num_iters = 0
    found_marker = None

    for marker in markers:
        count = sum(1 for e in data.get('traceEvents', []) if marker in e.get('name', ''))
        if count > 0:
            # Heuristic: if it's a kernel count, we might need to divide by expected calls per step
            # For now, we assume markers like 'fwdbwd' are 1-per-step
            dynamic_num_iters, found_marker = count, marker
            break
            
    if dynamic_num_iters == 0:
        raise ValueError("No iteration markers found in trace.")

    # Find the hardware span
    all_ts = []
    for path in kineto_files:
        with open(path, "r") as f:
            d = json.load(f)
            all_ts.extend([e['ts'] for e in d.get('traceEvents', []) if e.get('cat') == 'kernel'])
    
    iter_time_sec = (max(all_ts) - min(all_ts)) / dynamic_num_iters / 1e6
    return iter_time_sec, len(kineto_files), dynamic_num_iters, found_marker

def metric_cal(directory: str) -> float:
    # Look for the .sh script in the trace directory
    script_path = None
    for f in os.listdir(directory):
        if f.endswith(".sh"):
            script_path = os.path.join(directory, f)
            break
            
    if not script_path:
        raise FileNotFoundError(f"No .sh submit script found in {directory} to extract parameters.")

    params = _parse_shell_script(script_path)
    iter_time_sec, num_gpus, actual_num_iters, marker = _global_iteration_time_seconds(directory)
    
    flops_iter_global = _flops_per_iteration_global(params)

    print(f"--- MFU Tool Logic ---")
    print(f"Source Script: {os.path.basename(script_path)}")
    print(f"Params: B={params['B']}, S={params['S']}, L={params['L']}, H={params['H']}")
    print(f"Iteration Marker: '{marker}' ({actual_num_iters} iters)")
    print(f"Avg Iter Time: {iter_time_sec*1000:.3f} ms")

    denom = PEAK_FLOPS_PER_GPU * num_gpus * iter_time_sec
    return round(flops_iter_global / denom, 4)