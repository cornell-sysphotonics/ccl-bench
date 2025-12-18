#!/usr/bin/env python3
import argparse
import csv
import io
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

REPORT = "cuda_gpu_kern_sum"

# -----------------------------
# Bucket regexes (tune as needed)
# -----------------------------
# COMM: NCCL kernels + optional sendrecv keywords
COMM_REGEXES = [
    re.compile(r"^nccl", re.IGNORECASE),  # ncclDevKernel_* / ncclKernel_*
    re.compile(r"sendrecv", re.IGNORECASE),
    re.compile(r"allgather|reducescatter|allreduce|broadcast|reduce", re.IGNORECASE),
]

# ATTENTION: flash attention / softmax kernels often used by attention / KV cache ops
ATTN_REGEXES = [
    re.compile(r"flash", re.IGNORECASE),
    re.compile(r"fmha", re.IGNORECASE),
    re.compile(r"attention", re.IGNORECASE),
    re.compile(r"paged", re.IGNORECASE),          # vLLM paged attention/cache
    re.compile(r"kv", re.IGNORECASE),             # kv cache related (coarse)
    re.compile(r"reshape.*cache", re.IGNORECASE),
    re.compile(r"triton_.*softmax", re.IGNORECASE),  # often attention softmax (heuristic)
]

# MOE ROUTING / DISPATCH: topk, routing, alignment/packing/reordering
MOE_ROUTE_REGEXES = [
    re.compile(r"\b(topk|sbtopk|gathertopk)\b", re.IGNORECASE),
    re.compile(r"\b(router|routing|gate|gating)\b", re.IGNORECASE),
    re.compile(r"\b(moe_.*align|align_block|dispatch|scatter|gather|pack|reorder|sort|bucket|hist|prefix)\b", re.IGNORECASE),
    re.compile(r"\b(token.*expert)\b", re.IGNORECASE),
]

# MOE EXPERT GEMM / FUSED EXPERT: fused moe kernels or grouped expert kernels
MOE_EXPERT_REGEXES = [
    re.compile(r"\bfused[_\-]?moe\b", re.IGNORECASE),
    re.compile(r"\bexpert\b", re.IGNORECASE),
    re.compile(r"_fwd_grouped_kernel", re.IGNORECASE),
    re.compile(r"\bgrouped\b", re.IGNORECASE),
    re.compile(r"\bmoe\b", re.IGNORECASE),  # broad; keep last
    re.compile(r"\bgemm\b", re.IGNORECASE), # WARNING: GEMM also used by attention/MLP; we only count as expert if other moe clues match.
]

# If a kernel matches "gemm" but not moe clues, we should not classify it as expert gemm.
# We’ll treat "gemm" as expert only if kernel name ALSO matches one of these moe-hint patterns.
MOE_HINT_FOR_GEMM = [
    re.compile(r"\b(moe|expert|grouped|fused[_\-]?moe)\b", re.IGNORECASE),
]


def _run(cmd: List[str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def _run_nsys_kernsum_csv(trace_path: str) -> str:
    # NOTE: passing .sqlite is OK if already exported by nsys
    cmd = ["nsys", "stats", "--report", REPORT, "--format", "csv", trace_path]
    return _run(cmd)


def _extract_csv_block(nsys_stdout: str) -> List[str]:
    lines = [ln.rstrip("\n") for ln in nsys_stdout.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        # typical header:
        # Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
        if ln.startswith("Time (%)") and "Total Time" in ln and "Instances" in ln and "Name" in ln:
            header_idx = i
            break
    if header_idx is None:
        preview = "\n".join(lines[:50])
        raise RuntimeError(
            f"Could not find {REPORT} CSV header in nsys output.\nFirst ~50 lines:\n\n{preview}"
        )
    return lines[header_idx:]


def _is_match(name: str, regexes: List[re.Pattern]) -> bool:
    return any(r.search(name) for r in regexes)


def _classify_kernel(name: str) -> str:
    """
    Return bucket name among:
      COMM, ATTN, MOE_ROUTE, MOE_EXPERT, OTHER
    Priority matters (COMM first, etc).
    """
    # 1) communication
    if _is_match(name, COMM_REGEXES):
        return "COMM"

    # 2) attention
    if _is_match(name, ATTN_REGEXES):
        return "ATTN"

    # 3) moe routing/dispatch
    if _is_match(name, MOE_ROUTE_REGEXES):
        return "MOE_ROUTE"

    # 4) moe expert compute
    if _is_match(name, MOE_EXPERT_REGEXES):
        # Special-case GEMM: only count as MOE_EXPERT if it also has MOE hints
        if re.search(r"\bgemm\b", name, re.IGNORECASE):
            if not _is_match(name, MOE_HINT_FOR_GEMM):
                return "OTHER"
        return "MOE_EXPERT"

    return "OTHER"


def _parse_kernsum(csv_lines: List[str]) -> Dict[str, float]:
    """
    Parse csv and return bucket_ns dict + total_ns.
    Output keys: TOTAL_NS, COMM_NS, ATTN_NS, MOE_ROUTE_NS, MOE_EXPERT_NS, OTHER_NS
    """
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))

    # Basic sanity on columns (Nsight versions might vary slightly)
    fieldnames = reader.fieldnames or []
    if "Total Time (ns)" not in fieldnames or "Name" not in fieldnames:
        raise RuntimeError(f"Unexpected CSV columns: {fieldnames}")

    bucket_ns = {
        "COMM_NS": 0.0,
        "ATTN_NS": 0.0,
        "MOE_ROUTE_NS": 0.0,
        "MOE_EXPERT_NS": 0.0,
        "OTHER_NS": 0.0,
    }
    total_ns = 0.0

    for row in reader:
        name = row.get("Name", "")
        t_ns_s = row.get("Total Time (ns)", "")
        if not name or not t_ns_s:
            continue
        try:
            t_ns = float(t_ns_s)
        except ValueError:
            continue

        total_ns += t_ns
        bucket = _classify_kernel(name)

        if bucket == "COMM":
            bucket_ns["COMM_NS"] += t_ns
        elif bucket == "ATTN":
            bucket_ns["ATTN_NS"] += t_ns
        elif bucket == "MOE_ROUTE":
            bucket_ns["MOE_ROUTE_NS"] += t_ns
        elif bucket == "MOE_EXPERT":
            bucket_ns["MOE_EXPERT_NS"] += t_ns
        else:
            bucket_ns["OTHER_NS"] += t_ns

    bucket_ns["TOTAL_NS"] = total_ns
    return bucket_ns


def _find_traces(trace_dir: str) -> List[str]:
    if os.path.isfile(trace_dir):
        return [trace_dir]
    if not os.path.isdir(trace_dir):
        raise ValueError(f"Invalid path: {trace_dir}")

    reps = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir) if f.endswith(".nsys-rep")]
    sqls = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir) if f.endswith(".sqlite")]
    reps.sort()
    sqls.sort()
    return reps if reps else sqls


def _fmt_ms(ns: float) -> float:
    return ns / 1e6


def _fmt_pct(part: float, total: float) -> float:
    return (100.0 * part / total) if total > 0 else 0.0


def compute_breakdown(trace_dir: str) -> Dict[str, Dict[str, float]]:
    traces = _find_traces(trace_dir)
    if not traces:
        raise RuntimeError(f"No .nsys-rep or .sqlite found in: {trace_dir}")

    per_trace: Dict[str, Dict[str, float]] = {}
    for p in traces:
        out = _run_nsys_kernsum_csv(p)
        csv_lines = _extract_csv_block(out)
        ns = _parse_kernsum(csv_lines)

        total = ns["TOTAL_NS"]
        row = {
            "total_ms": _fmt_ms(total),

            "comm_ms": _fmt_ms(ns["COMM_NS"]),
            "comm_pct": _fmt_pct(ns["COMM_NS"], total),

            "attn_ms": _fmt_ms(ns["ATTN_NS"]),
            "attn_pct": _fmt_pct(ns["ATTN_NS"], total),

            "moe_route_ms": _fmt_ms(ns["MOE_ROUTE_NS"]),
            "moe_route_pct": _fmt_pct(ns["MOE_ROUTE_NS"], total),

            "moe_expert_ms": _fmt_ms(ns["MOE_EXPERT_NS"]),
            "moe_expert_pct": _fmt_pct(ns["MOE_EXPERT_NS"], total),

            "other_ms": _fmt_ms(ns["OTHER_NS"]),
            "other_pct": _fmt_pct(ns["OTHER_NS"], total),
        }
        per_trace[os.path.basename(p)] = row

    return per_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True, help="Directory containing .nsys-rep/.sqlite traces (or a single trace file)")
    ap.add_argument("--no-aggregate", action="store_true", help="Do not print aggregate summary")
    args = ap.parse_args()

    per_trace = compute_breakdown(args.trace_dir)

    # Print per-trace
    print("=" * 100)
    print(f"{'TRACE':40s}  {'TOTAL(ms)':>10s}  {'COMM%':>7s} {'ATTN%':>7s} {'ROUTE%':>7s} {'EXPERT%':>8s} {'OTHER%':>7s}")
    print("-" * 100)

    # aggregate sums in ns via ms * 1e6 is fine (we’ll just rescale)
    agg = {
        "total_ms": 0.0,
        "comm_ms": 0.0,
        "attn_ms": 0.0,
        "moe_route_ms": 0.0,
        "moe_expert_ms": 0.0,
        "other_ms": 0.0,
    }

    for name in sorted(per_trace.keys()):
        r = per_trace[name]
        print(
            f"{name:40s}  {r['total_ms']:10.3f}  "
            f"{r['comm_pct']:6.2f}% {r['attn_pct']:6.2f}% {r['moe_route_pct']:6.2f}% {r['moe_expert_pct']:7.2f}% {r['other_pct']:6.2f}%"
        )
        for k in agg:
            agg[k] += r[k]

    print("-" * 100)

    if not args.no_aggregate:
        tot = agg["total_ms"]
        def pct(x): return (100.0 * x / tot) if tot > 0 else 0.0
        print(
            f"{'AGGREGATE':40s}  {tot:10.3f}  "
            f"{pct(agg['comm_ms']):6.2f}% {pct(agg['attn_ms']):6.2f}% {pct(agg['moe_route_ms']):6.2f}% {pct(agg['moe_expert_ms']):7.2f}% {pct(agg['other_ms']):6.2f}%"
        )

    print("=" * 100)

    # Also print detailed ms per bucket for each trace (useful for tables)
    print("\nPer-trace times (ms):")
    for name in sorted(per_trace.keys()):
        r = per_trace[name]
        print(
            f"- {name}: total={r['total_ms']:.3f} | "
            f"comm={r['comm_ms']:.3f}, attn={r['attn_ms']:.3f}, "
            f"moe_route={r['moe_route_ms']:.3f}, moe_expert={r['moe_expert_ms']:.3f}, other={r['other_ms']:.3f}"
        )


if __name__ == "__main__":
    main()