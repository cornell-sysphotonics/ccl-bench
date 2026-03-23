"""
Shared trace-file utilities used by multiple metric tools.

Provides helpers for:
  - Opening plain or gzipped trace files
  - Finding trace files recursively in a directory
  - Extracting aggregate metrics (wall time, active FLOPs, bandwidth) from
    XLA / Chrome-trace JSON files
"""

import gzip
import json
import os
from typing import Dict, List, Optional, Tuple


# ── Type coercion helpers ─────────────────────────────────────────────────────

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


# ── Interval merging ─────────────────────────────────────────────────────────

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


# ── File I/O ─────────────────────────────────────────────────────────────────

def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def find_trace_files(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (run_dir_path, trace_file_path).
    Searches recursively for .trace.json or .json.gz files.
    """
    hits = []
    if os.path.isfile(root):
        if root.endswith(".trace.json") or root.endswith(".json.gz"):
            return [(os.path.dirname(root), root)]
        return []

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json") or fn.endswith(".trace.json.gz") or fn.endswith(".trace.json"):
                trace_path = os.path.join(dirpath, fn)
                run_dir = dirpath
                cur = dirpath
                while True:
                    base = os.path.basename(cur)
                    if base.startswith("MODEL_"):
                        run_dir = cur
                        break
                    parent = os.path.dirname(cur)
                    if parent == cur or not parent.startswith(root):
                        break
                    cur = parent
                hits.append((run_dir, trace_path))
    return hits


# ── Trace metric extraction ──────────────────────────────────────────────────

def extract_metrics_from_trace(trace_path: str, ts_unit: str = "us") -> Dict:
    """
    Extract aggregate metrics from a Chrome-trace JSON file.

    Returns dict with keys:
      wall_s, active_s, total_flops, total_bytes,
      bandwidth_wall_gbs, active_tflops
    """
    scale = 1e-6 if ts_unit == "us" else (1e-3 if ts_unit == "ms" else 1.0)

    with open_maybe_gz(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
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
