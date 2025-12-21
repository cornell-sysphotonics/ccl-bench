"""
Shared utilities for metric computation
"""
import os
import zipfile
import pandas as pd
import gzip
import json
import re
import numpy as np
from pathlib import Path
import math
from typing import Dict, Tuple, Optional


def upload_file(file_path: str):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if zipfile.is_zipfile(file_path):
        extract_dir = os.path.splitext(file_path)[0]
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"Extracted to: {os.path.abspath(extract_dir)}")
        return extract_dir
    else:
        abs_path = os.path.abspath(file_path)
        print("Not a zip file. File is ready to use at:")
        print(abs_path)
        return abs_path


# Helper functions
def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(x)
    except Exception:
        return default


def to_int(v):
    if isinstance(v, str) and v.isdigit():
        return int(v)
    if isinstance(v, (int, float, np.number)):
        return int(v)
    return None


def sum_model_flops_from_args(df: pd.DataFrame) -> float:
    total = 0.0
    if "args" not in df.columns:
        return 0.0
    for a in df["args"].values:
        if isinstance(a, dict) and "model_flops" in a:
            total += safe_float(a.get("model_flops"), default=0.0)
    return float(total)


# Functions for helping load traces
def load_chrome_trace_any(path: str):
    p = Path(path)
    
    abs_path = p.resolve() if p.exists() else p.absolute()
    
    if not p.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}\n"
            f"Absolute path attempted: {abs_path}\n"
            f"Current working directory: {os.getcwd()}"
        )
    
    if not p.is_file():
        raise ValueError(
            f"Path is not a file: {path}\n"
            f"Absolute path: {abs_path}\n"
            f"Is directory: {p.is_dir()}"
        )
    
    try:
        if p.suffix == ".gz" or str(p).endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from trace file {path}: {e}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading trace file {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading trace file {path} (absolute: {abs_path}): {type(e).__name__}: {e}")


def trace_events_to_df(trace) -> pd.DataFrame:
    evs = trace.get("traceEvents", trace)
    rows = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        if e.get("ph") != "X":
            continue
        if "ts" not in e or "dur" not in e:
            continue

        ts_us = float(e["ts"])
        dur_us = float(e["dur"])
        args = e.get("args", {})
        if not isinstance(args, dict):
            args = {}

        rows.append(
            {
                "name": e.get("name", ""),
                "cat": e.get("cat", ""),
                "pid": e.get("pid"),
                "tid": e.get("tid"),
                "ts_us": ts_us,
                "dur_us": dur_us,
                "start_s": ts_us / 1e6,
                "dur_s": dur_us / 1e6,
                "end_s": (ts_us + dur_us) / 1e6,
                "args": args,
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df["name_l"] = df["name"].astype(str).str.lower()
        df["cat_l"] = df["cat"].astype(str).str.lower()
    return df


def classify_row(name: str, cat: str) -> str:
    s = f"{name} {cat}".lower()
    if any(
        k in s
        for k in [
            "all-reduce",
            "allreduce",
            "all-gather",
            "allgather",
            "reduce-scatter",
            "reducescatter",
            "all-to-all",
            "alltoall",
            "collective",
            "nccl",
        ]
    ):
        return "comm"
    if any(k in s for k in ["dot", "convolution", "gemm", "matmul", "fusion", "compute", "custom-call"]):
        return "compute"
    return "other"


def add_kind_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kind"] = df.apply(lambda r: classify_row(r["name"], r["cat"]), axis=1)
    return df


# per event feature extraction
DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "s32": 4,
    "u32": 4,
    "s64": 8,
    "u64": 8,
    "pred": 1,
}
SHAPE_RE = re.compile(r"^([a-z0-9]+)\[([^\]]*)\]")


def device_duration_s(args: dict) -> float:
    v = to_int(args.get("device_duration_ps"))
    return (v / 1e12) if v is not None else np.nan


def payload_bytes_from_shape(args: dict) -> float:
    s = args.get("shape_with_layout")
    if not isinstance(s, str):
        return np.nan
    m = SHAPE_RE.match(s.strip())
    if not m:
        return np.nan
    dtype = m.group(1).lower()
    dims_str = m.group(2).strip()
    bpe = DTYPE_BYTES.get(dtype)
    if bpe is None:
        return np.nan
    dims = [int(x) for x in re.split(r"[,\s]+", dims_str) if x.strip().isdigit()]
    numel = int(np.prod(dims)) if dims else 1
    return float(numel * bpe)


def message_bytes(args: dict) -> float:
    b = payload_bytes_from_shape(args)
    if not np.isnan(b):
        return b
    for k in ["raw_bytes_accessed", "bytes_accessed"]:
        v = to_int(args.get(k))
        if v is not None and v > 0:
            return float(v)
    return np.nan


def model_flops(args: dict) -> float:
    v = args.get("model_flops")
    try:
        return float(v)
    except Exception:
        return np.nan


def add_event_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["T_s"] = df["args"].apply(device_duration_s)
    df["message_bytes"] = df["args"].apply(message_bytes)
    df["model_flops"] = df["args"].apply(model_flops)
    return df


def prepare_dataframe(trace_json_path: str) -> pd.DataFrame:
    """Load trace and prepare dataframe with all necessary columns."""
    trace = load_chrome_trace_any(trace_json_path)
    df = trace_events_to_df(trace)
    df = add_kind_column(df)
    df = add_event_feature_columns(df)
    return df

