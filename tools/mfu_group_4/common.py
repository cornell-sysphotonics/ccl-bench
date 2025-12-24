
import os
import gzip
import json
import re
from typing import List, Tuple, Optional, Dict

# ------------------ helpers ------------------

def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def to_int(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0
        try:
            return int(float(s))
        except ValueError:
            return 0
    return 0


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        last = merged[-1]
        if s <= last[1]:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def parse_run_dir(run_dir: str) -> dict:
    """
    Parse directory name.
    Supports strictly structured: MODEL_Qwen_Qwen3-4B,INPUT_1024,OUTPUT_1,BATCH_64,TP_8
    And fallback/loose parsing for things like: Qwen3-4B-torchxla-vllm-tp4-tpu-group-4
    """
    name = os.path.basename(run_dir.rstrip("/"))
    parts = [p.strip() for p in name.split(",")]

    out = {
        "run_dir_name": name,
        "model": None,
        "input_size": 0, # Default to 0 if not found
        "output_size": 0,
        "batch_size": 0,
        "tp": None,
    }
    
    # helper regex for TP
    tp_regex = re.compile(r"tp(\d+)", re.IGNORECASE)

    # Try strict parsing first
    strict_match = False
    for p in parts:
        if p.startswith("MODEL_"):
            out["model"] = p[len("MODEL_"):]
            strict_match = True
        elif p.startswith("INPUT_"):
            out["input_size"] = int(p.split("_", 1)[1])
        elif p.startswith("OUTPUT_"):
            out["output_size"] = int(p.split("_", 1)[1])
        elif p.startswith("BATCH_"):
            out["batch_size"] = int(p.split("_", 1)[1])
        elif p.startswith("TP_"):
            out["tp"] = int(p.split("_", 1)[1])

    if out["model"] and out["tp"] is not None:
        return out
        
    # Fallback / heuristic parsing if strict elements missing
    if not strict_match:
         # Assume name is model-like or look for components
         # Qwen3-4B-torchxla-vllm-tp4-tpu-group-4
         tokens = name.split("-")
         
         # extract TP
         # look for 'tp4' or 'tp=4'
         match = tp_regex.search(name)
         if match:
             out["tp"] = int(match.group(1))
         
         # model name is often the prefix
         out["model"] = name
         
    if out["tp"] is None:
         # Default to 1 if we really can't find it, or Error? 
         # The original script raised ValueError. 
         # Let's try to be robust but maybe log a warning if we had a logger.
         # For now, if we can't find TP, we might fail MFU calc, but let's see.
         pass

    return out


def find_trace_files(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (run_dir_path, trace_file_path)
    Search recursively for .trace.json or .trace.json.gz
    """
    hits = []
    # If root is a file, return it if it matches
    if os.path.isfile(root):
        if root.endswith(".trace.json") or root.endswith(".trace.json.gz"):
             return [(os.path.dirname(root), root)]
        return []

    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            # match "....trace.json" or "....trace.json.gz"
            if fn.endswith(".trace.json") or fn.endswith(".trace.json.gz"):
                trace_path = os.path.join(dirpath, fn)
                
                # Use the provided root or suitable parent as run_dir
                # The original script looked for MODEL_, but we might be in a loosely named dir.
                # We'll treat the immediate parent of the trace (or the root if shallow) as run_dir,
                # OR walk up to find a "meaningful" dir. 
                # For simplicity in this env, we default to the immediate parent or the passed root.
                
                run_dir = dirpath # standard fallback
                
                # Attempt to find "MODEL_" parent like original script
                cur = dirpath
                while True:
                    base = os.path.basename(cur)
                    if base.startswith("MODEL_"):
                        run_dir = cur
                        break
                    parent = os.path.dirname(cur)
                    if parent == cur or not parent.startswith(root): # Don't go above root
                        break
                    cur = parent
                
                hits.append((run_dir, trace_path))
    return hits


def extract_metrics_from_trace(trace_path: str, ts_unit: str = "us") -> dict:
    """
    ts_unit fixed to 'us' per your note: timestamps are microseconds.
    Computes:
      - wall_s (trace span)
      - active_s (union of intervals for events with model_flops>0)
      - total_flops (sum model_flops>0)
      - total_bytes (sum raw_bytes_accessed or bytes_accessed over events)
      - bandwidth_wall_gbs = total_bytes / wall_s / 1e9
      - active_tflops = (total_flops/1e12) / active_s
    """
    scale = 1e-6 if ts_unit == "us" else (1e-3 if ts_unit == "ms" else 1.0)

    with open_maybe_gz(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
         # It might be that the file is just a list of events not wrapped in `traceEvents` dict
         if isinstance(data, list):
             events = data
         else:
             raise ValueError(f"Unexpected trace format for: {trace_path}")

    min_ts = None
    max_te = None

    total_flops = 0
    total_bytes = 0

    compute_intervals = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") == "M":
            continue

        ts = to_float(ev.get("ts"))
        dur = to_float(ev.get("dur"))
        if ts is not None:
            if min_ts is None or ts < min_ts:
                min_ts = ts
            te = ts + (dur if dur is not None else 0.0)
            if max_te is None or te > max_te:
                max_te = te

        args = ev.get("args")
        if not isinstance(args, dict):
            continue

        b = to_int(args.get("raw_bytes_accessed") or args.get("bytes_accessed"))
        if b > 0:
            total_bytes += b

        fl = to_int(args.get("model_flops"))
        if fl > 0:
            total_flops += fl
            if ts is not None and dur is not None and dur > 0:
                compute_intervals.append((ts, ts + dur))

    if min_ts is None or max_te is None or max_te <= min_ts:
        # If we can't determine wall time, we can't compute bandwidth/flops correctly
        # return NaNs or 0
        return {
            "wall_s": 0.0,
            "active_s": 0.0,
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "bandwidth_wall_gbs": float("nan"),
            "active_tflops": float("nan"),
        }

    wall_s = (max_te - min_ts) * scale

    merged = merge_intervals(compute_intervals)
    active_raw = sum(e - s for s, e in merged)
    active_s = active_raw * scale

    bandwidth_wall_gbs = (total_bytes / 1e9) / wall_s if wall_s > 0 else float("nan")
    active_tflops = ((total_flops / 1e12) / active_s) if active_s > 0 else float("nan")

    return {
        "wall_s": wall_s,
        "active_s": active_s,
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "bandwidth_wall_gbs": bandwidth_wall_gbs,
        "active_tflops": active_tflops,
    }
