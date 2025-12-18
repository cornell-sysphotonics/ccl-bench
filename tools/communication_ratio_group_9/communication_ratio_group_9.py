#!/usr/bin/env python3
import argparse
import csv
import io
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


# ---- Kernel-name rules for what counts as "communication" (extend as needed) ----
# 1) NCCL GPU kernels
# 2) Optional backend-specific cross-device reduce/ops (if you want to include them)
COMM_REGEXES = [
    re.compile(r"^nccl", re.IGNORECASE),  # ncclDevKernel_* / ncclKernel_*
    re.compile(r"sendrecv", re.IGNORECASE),  # some NCCL names include SendRecv
    re.compile(
        r"cross[_\-]?device", re.IGNORECASE
    ),  # optional: cross-device reduce/ops
    re.compile(r"allgather|reducescatter|allreduce|broadcast|reduce", re.IGNORECASE),
]


@dataclass
class TraceStat:
    path: str
    total_ns: int
    comm_ns: int
    comm_calls: int  # approximate: sum of Instances for comm kernels

    @property
    def ratio(self) -> float:
        return (self.comm_ns / self.total_ns) if self.total_ns > 0 else 0.0


def _run_nsys_kernsum_csv(trace_path: str) -> str:
    """
    Run:
      nsys stats --report cuda_gpu_kern_sum --format csv <trace>
    and return stdout text (CSV may be preceded by 'Processing...' lines).
    """
    cmd = [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        trace_path,
    ]
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
            "Cannot find 'nsys' in PATH. Try 'which nsys' and load the Nsight Systems module."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"nsys stats failed for {trace_path}\n--- output ---\n{e.stdout}"
        ) from e


def _extract_csv_block(nsys_stdout: str) -> List[str]:
    """
    nsys stdout often looks like:
      Processing [...] with [cuda_gpu_kern_sum.py]...
      Time (%),Total Time (ns),Instances,...
      ...
    We locate the header line starting with 'Time (%)' and return all lines from there.
    """
    lines = [ln.strip("\n") for ln in nsys_stdout.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        # Robust match for the header
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


def _parse_kernsum_csv_lines(csv_lines: List[str]) -> Tuple[int, int, int]:
    """
    Return (total_ns, comm_ns, comm_calls).
    """
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))

    # Expected columns:
    # Time (%),Total Time (ns),Instances,Avg (ns),...,Name
    # Some Nsight versions may rename columns; try a few candidates.
    def get_col(d, *cands):
        for c in cands:
            if c in d:
                return d[c]
        return None

    total_ns = 0
    comm_ns = 0
    comm_calls = 0

    for row in reader:
        t_ns_s = get_col(row, "Total Time (ns)", "Total Time (nsec)", "Total Time")
        inst_s = get_col(row, "Instances", "Count")
        name = get_col(row, "Name", "Kernel Name", "Function")

        if t_ns_s is None or name is None:
            continue

        try:
            t_ns = int(float(t_ns_s))
        except ValueError:
            continue

        total_ns += t_ns

        if _is_comm_kernel(name):
            comm_ns += t_ns
            if inst_s is not None:
                try:
                    comm_calls += int(float(inst_s))
                except ValueError:
                    pass

    return total_ns, comm_ns, comm_calls


def compute_comm_ratio(trace_dir: str) -> float:
    # Collect .nsys-rep and .sqlite traces
    paths = []
    for fn in os.listdir(trace_dir):
        if fn.endswith(".nsys-rep") or fn.endswith(".sqlite"):
            paths.append(os.path.join(trace_dir, fn))
    paths.sort()

    if not paths:
        raise RuntimeError(f"No .nsys-rep or .sqlite traces found in: {trace_dir}")

    stats: List[TraceStat] = []
    for pth in paths:
        out = _run_nsys_kernsum_csv(pth)
        csv_lines = _extract_csv_block(out)
        total_ns, comm_ns, comm_calls = _parse_kernsum_csv_lines(csv_lines)
        stats.append(
            TraceStat(
                path=pth, total_ns=total_ns, comm_ns=comm_ns, comm_calls=comm_calls
            )
        )

    # Weighted aggregate = sum(comm) / sum(total)
    total_ns_sum = sum(s.total_ns for s in stats)
    comm_ns_sum = sum(s.comm_ns for s in stats)
    ratio = (comm_ns_sum / total_ns_sum) if total_ns_sum > 0 else 0.0

    # Print per-trace breakdown (sanity check)
    print("=" * 80)
    for s in stats:
        print(
            f"{os.path.basename(s.path):40s}  comm_ratio={s.ratio * 100:6.2f}%"
            f"  total={s.total_ns / 1e6:10.3f} ms  comm={s.comm_ns / 1e6:10.3f} ms"
            f"  comm_calls~{s.comm_calls}"
        )
    print("-" * 80)
    print(f"AGGREGATE comm_ratio = {ratio:.6f}  ({ratio * 100:.2f}%)")
    print("=" * 80)

    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing .nsys-rep/.sqlite traces",
    )
    args = ap.parse_args()
    compute_comm_ratio(args.trace_dir)


if __name__ == "__main__":
    main()
