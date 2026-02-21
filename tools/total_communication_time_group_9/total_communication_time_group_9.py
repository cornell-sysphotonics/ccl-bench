#!/usr/bin/env python3
import argparse
import csv
import io
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List

REPORT = "cuda_gpu_kern_sum"

# ---- Kernel-name rules for what counts as communication
#      (kept consistent with communication_ratio.py) ----
COMM_REGEXES = [
    re.compile(r"^nccl", re.IGNORECASE),
    re.compile(r"sendrecv", re.IGNORECASE),
    re.compile(
        r"cross[_\-]?device", re.IGNORECASE
    ),  # optional: cross-device reduce/ops
    re.compile(r"allgather|reducescatter|allreduce|broadcast|reduce", re.IGNORECASE),
]


@dataclass
class TraceComm:
    path: str
    comm_ns: int

    @property
    def comm_ms(self) -> float:
        return self.comm_ns / 1e6


def _run_nsys_kernsum_csv(trace_path: str) -> str:
    cmd = ["nsys", "stats", "--report", REPORT, "--format", "csv", trace_path]
    try:
        p = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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


def _extract_csv_block(nsys_stdout: str) -> List[str]:
    lines = [ln.strip("\n") for ln in nsys_stdout.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if (
            ln.startswith("Time (%)")
            and "Total Time" in ln
            and "Instances" in ln
            and "Name" in ln
        ):
            header_idx = i
            break
    if header_idx is None:
        preview = "\n".join(lines[:40])
        raise RuntimeError(
            "Could not locate cuda_gpu_kern_sum CSV header in nsys output.\n"
            "First ~40 lines:\n\n" + preview
        )
    return lines[header_idx:]


def _is_comm_kernel(name: str) -> bool:
    return any(r.search(name) for r in COMM_REGEXES)


def _parse_comm_ns(csv_lines: List[str]) -> int:
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))

    def get_col(d, *cands):
        for c in cands:
            if c in d:
                return d[c]
        return None

    comm_ns = 0
    for row in reader:
        t_ns_s = get_col(row, "Total Time (ns)", "Total Time (nsec)", "Total Time")
        name = get_col(row, "Name", "Kernel Name", "Function")
        if t_ns_s is None or name is None:
            continue

        try:
            t_ns = int(float(t_ns_s))
        except ValueError:
            continue

        if _is_comm_kernel(name):
            comm_ns += t_ns

    return comm_ns


def _collect_traces(trace_dir: str) -> List[str]:
    if os.path.isfile(trace_dir):
        if trace_dir.endswith(".nsys-rep") or trace_dir.endswith(".sqlite"):
            return [trace_dir]
        raise ValueError(
            f"Unsupported trace type: {trace_dir} (expected .nsys-rep or .sqlite)"
        )

    if not os.path.isdir(trace_dir):
        raise ValueError(f"Invalid trace path: {trace_dir}")

    paths = []
    for fn in os.listdir(trace_dir):
        if fn.endswith(".nsys-rep") or fn.endswith(".sqlite"):
            paths.append(os.path.join(trace_dir, fn))
    paths.sort()
    if not paths:
        raise RuntimeError(f"No .nsys-rep or .sqlite traces found in: {trace_dir}")
    return paths


def compute_total_comm_time(trace_dir: str) -> Dict[str, float]:
    """
    Input:
      trace_dir (directory) OR a single trace file.

    Output:
      dict { trace_basename -> communication kernel time (ms) }
    """
    traces = _collect_traces(trace_dir)
    results: Dict[str, float] = {}

    for pth in traces:
        out = _run_nsys_kernsum_csv(pth)
        csv_lines = _extract_csv_block(out)
        comm_ns = _parse_comm_ns(csv_lines)
        results[os.path.basename(pth)] = comm_ns / 1e6

    # Print per-trace results for sanity checking
    print("=" * 80)
    for k, v in results.items():
        print(f"{k:40s}  comm_kernel_time={v:12.3f} ms")
    print("=" * 80)

    return results


# Optional alias if the framework expects `metric_cal`
def metric_cal(trace_dir: str):
    return compute_total_comm_time(trace_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing .nsys-rep/.sqlite traces (or a single trace file)",
    )
    args = ap.parse_args()
    compute_total_comm_time(args.trace_dir)


if __name__ == "__main__":
    main()
