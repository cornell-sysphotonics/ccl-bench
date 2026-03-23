"""
Shared utilities for NSYS trace processing.

Two approaches are used across metrics:
  1. SQLite — query CUPTI_ACTIVITY_KIND_KERNEL table directly
  2. CLI    — run `nsys stats --report cuda_gpu_kern_sum --format csv`
"""

import csv
import io
import os
import subprocess
import sys


# ── SQLite helpers ────────────────────────────────────────────────────────────

def find_sqlite_file(path: str):
    """Find .sqlite file in directory or return path if it's already a .sqlite file."""
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.endswith(".sqlite"):
        return path
    if os.path.isdir(path):
        sqlite_files = [f for f in os.listdir(path) if f.endswith(".sqlite")]
        if not sqlite_files:
            return None
        non_profiling = [f for f in sqlite_files if "profiling" not in f.lower()]
        if non_profiling:
            return os.path.abspath(os.path.join(path, non_profiling[0]))
        return os.path.abspath(os.path.join(path, sqlite_files[0]))
    return None


# ── NSYS CLI helpers ──────────────────────────────────────────────────────────

REPORT = "cuda_gpu_kern_sum"


def run_nsys_kernsum_csv(trace_path: str) -> str:
    """Run nsys stats --report cuda_gpu_kern_sum --format csv and return stdout."""
    cmd = ["nsys", "stats", "--report", REPORT, "--format", "csv", trace_path]
    try:
        p = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return p.stdout
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot find 'nsys' in PATH. Try 'which nsys' or load the Nsight Systems module."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"nsys stats failed for {trace_path}\n--- output ---\n{e.stdout}"
        ) from e


def extract_csv_block(nsys_stdout: str) -> list:
    """Locate the CSV header line and return all lines from header onward."""
    lines = [ln.strip("\n") for ln in nsys_stdout.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if (
            ln.startswith("Time (%)")
            and "Total Time" in ln
            and "Instances" in ln
            and "Name" in ln
        ):
            return lines[i:]
    preview = "\n".join(lines[:40])
    raise RuntimeError(
        "Could not locate cuda_gpu_kern_sum CSV header in nsys output.\n"
        "First ~40 lines:\n\n" + preview
    )


def _csv_col(row: dict, *candidates):
    """Return first matching column value from a CSV row."""
    for c in candidates:
        if c in row:
            return row[c]
    return None


def parse_kernsum_csv(csv_lines: list) -> list:
    """Parse CSV lines into list of dicts with keys: name, total_ns, instances."""
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        t_ns_s = _csv_col(row, "Total Time (ns)", "Total Time (nsec)", "Total Time")
        name = _csv_col(row, "Name", "Kernel Name", "Function")
        inst_s = _csv_col(row, "Instances", "Count")
        if t_ns_s is None or name is None:
            continue
        try:
            t_ns = int(float(t_ns_s))
        except ValueError:
            continue
        instances = 0
        if inst_s is not None:
            try:
                instances = int(float(inst_s))
            except ValueError:
                pass
        rows.append({"name": name, "total_ns": t_ns, "instances": instances})
    return rows


def collect_nsys_traces(trace_dir: str) -> list:
    """Collect .nsys-rep and .sqlite file paths from a directory (or single file)."""
    if os.path.isfile(trace_dir):
        if trace_dir.endswith(".nsys-rep") or trace_dir.endswith(".sqlite"):
            return [trace_dir]
        raise ValueError(f"Unsupported trace type: {trace_dir}")
    if not os.path.isdir(trace_dir):
        raise ValueError(f"Invalid trace path: {trace_dir}")
    paths = sorted(
        os.path.join(trace_dir, fn)
        for fn in os.listdir(trace_dir)
        if fn.endswith(".nsys-rep") or fn.endswith(".sqlite")
    )
    if not paths:
        raise RuntimeError(f"No .nsys-rep or .sqlite found in: {trace_dir}")
    return paths
