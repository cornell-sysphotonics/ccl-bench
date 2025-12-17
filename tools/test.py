#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

# NCCL kernel name prefixes commonly seen in gpukernsum
NCCL_PREFIXES = (
    "ncclDevKernel_",
    "ncclKernel_",  # sometimes appears
)

# Match common collectives if you want to restrict further
NCCL_COLLECTIVE_PAT = re.compile(
    r"^(ncclDevKernel_|ncclKernel_)(AllReduce|ReduceScatter|AllGather|Broadcast|Reduce|SendRecv)",
    re.IGNORECASE,
)


@dataclass
class KernelRow:
    name: str
    instances: int
    total_time_us: float  # microseconds


def _find_nsys_rep(trace_path: str) -> str:
    if os.path.isfile(trace_path) and trace_path.endswith(".nsys-rep"):
        return trace_path
    if os.path.isdir(trace_path):
        reps = [
            os.path.join(trace_path, f)
            for f in os.listdir(trace_path)
            if f.endswith(".nsys-rep")
        ]
        if not reps:
            raise FileNotFoundError(f"No .nsys-rep found under directory: {trace_path}")
        reps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return reps[0]
    raise FileNotFoundError(f"Trace path not found: {trace_path}")


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"stdout:\n{p.stdout}\n\n"
            f"stderr:\n{p.stderr}\n"
        )


def _export_gpukernsum_csv(nsys_rep: str, out_csv: str) -> None:
    """
    Export gpukernsum report to CSV.
    Notes:
      - nsys 'stats' uses '-o <basename>' and appends suffixes; however with '--format csv'
        it will produce a .csv for the report. We handle this by exporting into a temp dir
        and then locating the csv.
    """
    if shutil.which("nsys") is None:
        raise RuntimeError(
            "nsys not found in PATH. Please load Nsight Systems module or add nsys to PATH."
        )

    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "report")
        cmd = [
            "nsys",
            "stats",
            "--report",
            "gpukernsum",
            "--format",
            "csv",
            "-o",
            base,
            nsys_rep,
        ]
        _run(cmd)

        # Find produced CSV (nsys naming can vary by version)
        candidates = []
        for f in os.listdir(td):
            if f.endswith(".csv") and "gpukernsum" in f:
                candidates.append(os.path.join(td, f))
        if not candidates:
            # fallback: any csv
            candidates = [
                os.path.join(td, f) for f in os.listdir(td) if f.endswith(".csv")
            ]

        if not candidates:
            raise RuntimeError(
                f"nsys stats produced no CSV in {td}. Check nsys version / permissions."
            )

        candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        src = candidates[0]
        shutil.copyfile(src, out_csv)


def _parse_gpukernsum_csv(path: str) -> List[KernelRow]:
    """
    Parse gpukernsum CSV.

    gpukernsum CSV columns vary; we look for:
      - Name (kernel name)
      - Instances
      - Total Time (us) or Total Time or Time (us)
    """
    rows: List[KernelRow] = []

    with open(path, "r", newline="") as f:
        # Some nsys CSVs include metadata/header lines before the actual table.
        # We'll scan until we find a header row containing "Name" and "Instances".
        all_lines = f.readlines()

    header_idx = None
    header = None
    for i, line in enumerate(all_lines):
        if (
            "Name" in line
            and "Instances" in line
            and ("Total" in line or "Time" in line)
        ):
            header_idx = i
            header = [h.strip() for h in next(csv.reader([line]))]
            break

    if header_idx is None or header is None:
        raise RuntimeError(
            "Could not find gpukernsum table header in CSV. "
            "Open the CSV and check its format; you may need to adjust column matching."
        )

    # Build a DictReader starting at the header line
    table_text = "".join(all_lines[header_idx:])
    reader = csv.DictReader(table_text.splitlines())

    # Find best-matching columns
    fieldnames = reader.fieldnames or []
    name_col = _pick_col(fieldnames, ["Name", "Kernel Name", "Kernel"])
    inst_col = _pick_col(fieldnames, ["Instances", "Instance"])
    total_us_col = _pick_col(
        fieldnames, ["Total Time (us)", "Total Time", "Time (us)", "Total (us)"]
    )

    if not (name_col and inst_col and total_us_col):
        raise RuntimeError(
            "Could not locate required columns in gpukernsum CSV.\n"
            f"Found columns: {fieldnames}\n"
            f"Picked: name={name_col}, instances={inst_col}, total={total_us_col}"
        )

    for r in reader:
        name = (r.get(name_col) or "").strip()
        if not name:
            continue

        inst = _to_int(r.get(inst_col))
        total_us = _to_float(r.get(total_us_col))

        # Skip bogus rows
        if inst is None or total_us is None:
            continue

        rows.append(KernelRow(name=name, instances=inst, total_time_us=total_us))

    return rows


def _pick_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    # exact match first
    for c in candidates:
        if c in fieldnames:
            return c
    # contains match fallback
    for f in fieldnames:
        for c in candidates:
            if c.lower() in f.lower():
                return f
    return None


def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def compute_metrics(rows: List[KernelRow], restrict_collectives: bool) -> dict:
    total_gpu_us = sum(r.total_time_us for r in rows)

    def is_nccl(name: str) -> bool:
        if not name.startswith(NCCL_PREFIXES):
            return False
        if restrict_collectives:
            return NCCL_COLLECTIVE_PAT.match(name) is not None
        return True

    nccl_rows = [r for r in rows if is_nccl(r.name)]
    comm_us = sum(r.total_time_us for r in nccl_rows)
    coll_calls = sum(r.instances for r in nccl_rows)

    # top NCCL kernels by total time
    top_nccl = sorted(nccl_rows, key=lambda r: r.total_time_us, reverse=True)[:10]

    return {
        "coll_call_num": coll_calls,
        "comm_time_s": comm_us / 1e6,
        "total_gpu_kernel_time_s": total_gpu_us / 1e6,
        "comm_ratio": (comm_us / total_gpu_us) if total_gpu_us > 0 else 0.0,
        "top_nccl": [(r.name, r.instances, r.total_time_us / 1e6) for r in top_nccl],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace",
        required=True,
        help="Path to a .nsys-rep file OR a directory containing .nsys-rep",
    )
    ap.add_argument(
        "--restrict-collectives",
        action="store_true",
        help="Only count common collectives (AllReduce/AllGather/ReduceScatter/Broadcast/Reduce/SendRecv)",
    )
    ap.add_argument(
        "--keep-csv",
        action="store_true",
        help="Keep the exported gpukernsum CSV next to the trace",
    )
    args = ap.parse_args()

    rep = _find_nsys_rep(args.trace)
    out_dir = os.path.dirname(rep)
    out_csv = os.path.join(out_dir, "gpukernsum_export.csv")

    _export_gpukernsum_csv(rep, out_csv)
    rows = _parse_gpukernsum_csv(out_csv)
    m = compute_metrics(rows, restrict_collectives=args.restrict_collectives)

    if not args.keep_csv:
        try:
            os.remove(out_csv)
        except OSError:
            pass

    print(f"nsys_rep: {rep}")
    print(f"coll_call_num: {m['coll_call_num']}")
    print(f"comm_time_s: {m['comm_time_s']:.6f}")
    print(f"total_gpu_kernel_time_s: {m['total_gpu_kernel_time_s']:.6f}")
    print(f"comm_ratio: {m['comm_ratio'] * 100:.2f}%")
    print("\nTop NCCL kernels by total time:")
    for name, inst, tsec in m["top_nccl"]:
        print(f"  {tsec:10.6f}s  inst={inst:8d}  {name}")


if __name__ == "__main__":
    main()
