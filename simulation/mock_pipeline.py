#!/usr/bin/env python3
"""
Analytical simulation pipeline for CCL-Bench traces — no Docker or AstraSim required.

Reads rankN_trace.json files (kineto Chrome JSON), extracts NCCL collective
events, and estimates step time and comm fraction using the alpha-beta model
for the ring algorithm.  Accepts the same network parameters as pipeline.py
so results can be compared with full AstraSim runs.

Alpha-beta model (ring algorithm):
  AllReduce:            time = 2 × (n−1)/n × S/BW  +  2×(n−1) × lat
  AllGather / ReduceScatter / AllToAll:
                        time = (n−1)/n × S/BW  +  (n−1) × lat
  where BW is per-link bandwidth (bytes/ns = GB/s), lat is per-hop latency (ns).

Usage:
  python simulation/mock_pipeline.py --trace-dir /tmp/mock_trace [options]
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path


DTYPE_BYTES = {
    "BFloat16": 2, "BF16": 2,
    "Float16": 2, "Half": 2, "FP16": 2,
    "Float32": 4, "Float": 4, "FP32": 4,
    "Float64": 8, "Double": 8,
}

COLL_NAME_MAP = {
    "all_reduce": "all_reduce",
    "allreduce": "all_reduce",
    "_all_reduce": "all_reduce",
    "all_gather": "all_gather",
    "allgather": "all_gather",
    "_allgather_base": "all_gather",
    "all_gather_base": "all_gather",
    "allgather_into_tensor_coalesced": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "reducescatter": "reduce_scatter",
    "reduce_scatter_tensor": "reduce_scatter",
    "reduce_scatter_tensor_coalesced": "reduce_scatter",
    "all_to_all": "all_to_all",
    "all_to_allv": "all_to_all",
    "alltoall": "all_to_all",
}

KERNEL_RE_MAP = [
    (re.compile(r"AllReduce",     re.I), "all_reduce"),
    (re.compile(r"AllGather",     re.I), "all_gather"),
    (re.compile(r"ReduceScatter", re.I), "reduce_scatter"),
    (re.compile(r"AllToAll|SendRecv", re.I), "all_to_all"),
]


def _is_nccl_kernel(event: dict) -> bool:
    name = event.get("name", "")
    return event.get("cat") == "kernel" and (
        "ncclDev" in name or "ncclKernel" in name
    )


def _coll_type(event: dict) -> str | None:
    args = event.get("args", {})
    raw = args.get("Collective name", "").lower().strip()
    if raw in COLL_NAME_MAP:
        return COLL_NAME_MAP[raw]
    name = event.get("name", "")
    for pattern, ctype in KERNEL_RE_MAP:
        if pattern.search(name):
            return ctype
    return None


def _comm_size_bytes(event: dict) -> int:
    args = event.get("args", {})
    dtype = str(args.get("Data Type", "BFloat16"))
    elem_bytes = DTYPE_BYTES.get(dtype, 2)
    nelems = int(args.get("In msg nelems", 0) or 0)
    if nelems == 0:
        nelems = int(args.get("Out msg nelems", 0) or 0)
    return nelems * elem_bytes


def _parse_pg_ranks(raw) -> tuple[int, ...]:
    if not raw:
        return ()
    if isinstance(raw, (list, tuple)):
        try:
            return tuple(sorted(int(r) for r in raw))
        except (ValueError, TypeError):
            return ()
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, (list, tuple)):
            return tuple(sorted(int(r) for r in parsed))
    except (ValueError, SyntaxError):
        pass
    nums = re.findall(r"\d+", str(raw))
    return tuple(sorted(int(n) for n in nums)) if nums else ()


def load_rank_trace(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    return raw.get("traceEvents", raw.get("traceEvents", []))


def _choose_bandwidth(pg_ranks: tuple[int, ...], gpus_per_node: int,
                      intra_bw: float, inter_bw: float) -> float:
    """Use intra-node BW for groups confined to one node, else inter-node BW."""
    if not pg_ranks or gpus_per_node <= 0:
        return inter_bw
    nodes = {r // gpus_per_node for r in pg_ranks}
    return intra_bw if len(nodes) == 1 else inter_bw


def _ring_time_ns(coll_type: str, size_bytes: int, n: int,
                  bw_GBps: float, lat_ns: float) -> float:
    """Alpha-beta analytical comm time in nanoseconds for ring algorithm."""
    if n <= 1 or size_bytes <= 0:
        return 0.0
    # bw_GBps = bytes/ns  (1 GB/s = 1 byte/ns)
    bw = bw_GBps
    if coll_type == "all_reduce":
        return 2.0 * (n - 1) / n * size_bytes / bw + 2.0 * (n - 1) * lat_ns
    else:
        return (n - 1) / n * size_bytes / bw + (n - 1) * lat_ns


def simulate(trace_dir: Path, gpus_per_node: int,
             intra_bw: float, inter_bw: float,
             intra_lat: float, inter_lat: float) -> dict[int, dict[str, float]]:
    """
    Returns per-rank stats: {rank: {"wall_ns": float, "comm_ns": float, "compute_ns": float}}
    """
    trace_files = sorted(trace_dir.glob("rank*_trace.json"))
    if not trace_files:
        print(f"ERROR: no rank*_trace.json files found in {trace_dir}")
        sys.exit(1)

    ranks = []
    for f in trace_files:
        m = re.search(r"rank(\d+)_trace\.json", f.name)
        if m:
            ranks.append((int(m.group(1)), f))
    ranks.sort()

    results: dict[int, dict[str, float]] = {}

    for rank, path in ranks:
        events = load_rank_trace(path)

        # Separate NCCL collectives and non-NCCL GPU kernels
        nccl_events = []
        compute_events = []
        for e in events:
            if e.get("ph") != "X" or e.get("dur", 0) <= 0:
                continue
            if _is_nccl_kernel(e):
                nccl_events.append(e)
            elif e.get("cat") == "kernel":
                compute_events.append(e)

        # Compute: sum measured kernel durations
        compute_us = sum(e["dur"] for e in compute_events)

        # Communication: analytical time per collective
        comm_ns = 0.0
        for e in nccl_events:
            ctype = _coll_type(e)
            if ctype is None:
                continue
            size = _comm_size_bytes(e)
            pg_raw = e.get("args", {}).get("Process Group Ranks", "")
            pg = _parse_pg_ranks(pg_raw)
            bw = _choose_bandwidth(pg, gpus_per_node, intra_bw, inter_bw)
            lat = intra_lat if (
                pg and gpus_per_node > 0 and
                len({r // gpus_per_node for r in pg}) == 1
            ) else inter_lat
            n = len(pg) if pg else 2
            comm_ns += _ring_time_ns(ctype, size, n, bw, lat)

        compute_ns = compute_us * 1_000  # µs → ns
        wall_ns = compute_ns + comm_ns

        results[rank] = {
            "wall_ns":    wall_ns,
            "comm_ns":    comm_ns,
            "compute_ns": compute_ns,
        }

    return results


def show_results(stats: dict[int, dict[str, float]]):
    print("\n[mock_pipeline] Analytical simulation results (ns):")
    header = (f"  {'sys':>4}  {'Wall time':>16}  {'Comm time':>16}"
              f"  {'Compute time':>16}  {'Comm %':>7}")
    print(header)

    for sys_id in sorted(stats):
        s = stats[sys_id]
        wall = s["wall_ns"]
        comm = s["comm_ns"]
        comp = s["compute_ns"]
        pct = 100.0 * comm / wall if wall > 0 else 0.0
        print(f"  {sys_id:>4}  {wall:>16,.0f}  {comm:>16,.0f}"
              f"  {comp:>16,.0f}  {pct:>6.1f}%")

    if stats:
        walls = [s["wall_ns"] for s in stats.values()]
        comms = [s["comm_ns"] for s in stats.values()]
        avg_wall = sum(walls) / len(walls)
        avg_comm = sum(comms) / len(comms)
        avg_pct = 100.0 * avg_comm / avg_wall if avg_wall > 0 else 0.0
        print(f"  {'avg':>4}  {avg_wall:>16,.0f}  {avg_comm:>16,.0f}"
              f"  {'':>16}  {avg_pct:>6.1f}%")
        print(f"\n  Simulated step time: {avg_wall/1e6:.1f} ms"
              f"  |  Comm fraction: {avg_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analytical CCL-Bench simulation pipeline (no Docker/AstraSim)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline: 8-rank trace at A100-class NVLink + Slingshot
  python simulation/mock_pipeline.py --trace-dir /tmp/mock_trace

  # What-if: 2× scale-out bandwidth
  python simulation/mock_pipeline.py --trace-dir /tmp/mock_trace --bandwidth 400

  # What-if: H100-class intra-node (900 GB/s NVLink)
  python simulation/mock_pipeline.py --trace-dir /tmp/mock_trace --intra-bandwidth 900
""",
    )
    parser.add_argument("--trace-dir", required=True,
                        help="Directory containing rank*_trace.json files")
    parser.add_argument("--gpus-per-node", type=int, default=4,
                        help="Ranks per node (sets intra/inter group boundary, default: 4)")
    parser.add_argument("--intra-bandwidth", type=float, default=400.0,
                        help="Intra-node per-link bandwidth GB/s (default: 400, NVLink-class)")
    parser.add_argument("--intra-latency", type=float, default=50.0,
                        help="Intra-node per-link latency ns (default: 50)")
    parser.add_argument("--bandwidth", type=float, default=25.0,
                        help="Inter-node per-link bandwidth GB/s (default: 25, Slingshot HDR)")
    parser.add_argument("--latency", type=float, default=5000.0,
                        help="Inter-node per-link latency ns (default: 5000)")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.exists():
        print(f"ERROR: trace directory not found: {trace_dir}")
        sys.exit(1)

    print(f"[mock_pipeline] Trace dir: {trace_dir}")
    print(f"[mock_pipeline] Hardware:  intra={args.intra_bandwidth} GB/s / "
          f"{args.intra_latency} ns,  inter={args.bandwidth} GB/s / "
          f"{args.latency} ns,  gpus_per_node={args.gpus_per_node}")

    stats = simulate(
        trace_dir,
        gpus_per_node=args.gpus_per_node,
        intra_bw=args.intra_bandwidth,
        inter_bw=args.bandwidth,
        intra_lat=args.intra_latency,
        inter_lat=args.latency,
    )
    show_results(stats)
    print("\n[mock_pipeline] Complete.")


if __name__ == "__main__":
    main()
