#!/usr/bin/env python3
"""
Generate Chakra Execution Trace (.et) files from kineto-only rankN_trace.json files.

Runs inside the astra-sim Docker container where the chakra package is installed.
Extracts NCCL collective operations from each rank's trace and emits a sequence of
COMP_NODE (measured compute gap) + COMM_COLL_NODE (collective op) nodes per rank.

Process group information is extracted from the "Process Group Ranks" field of NCCL
kernel events and used to:
  - set pg_name attribute on COMM_COLL_NODE entries (required by astra-sim-hybrid-parallelism)
  - emit comm_group.json mapping unique group-id → participating NPU list

Usage (inside Docker):
    python3 gen_chakra_et.py --trace-dir /mnt/traces --output-dir /mnt/output --ranks 0,1,...
"""

import argparse
import ast
import json
from pathlib import Path

# Chakra is installed in the Docker venv at /opt/venv/astra-sim
from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
)
from chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

DTYPE_BYTES = {
    "BFloat16": 2, "BF16": 2,
    "Float16": 2, "Half": 2, "FP16": 2,
    "Float32": 4, "Float": 4, "FP32": 4,
    "Float64": 8, "Double": 8,
    "Int8": 1, "Int16": 2, "Int32": 4, "Int64": 8,
}

# Maps NCCL 'Collective name' field → Chakra collective type
COLL_NAME_MAP = {
    "all_reduce": ALL_REDUCE,
    "allreduce": ALL_REDUCE,
    "_all_reduce": ALL_REDUCE,
    "all_gather": ALL_GATHER,
    "allgather": ALL_GATHER,
    "_allgather_base": ALL_GATHER,
    "all_gather_base": ALL_GATHER,
    "allgather_into_tensor_coalesced": ALL_GATHER,
    "reduce_scatter": REDUCE_SCATTER,
    "reducescatter": REDUCE_SCATTER,
    "reduce_scatter_tensor": REDUCE_SCATTER,
    "reduce_scatter_tensor_coalesced": REDUCE_SCATTER,
    "all_to_all": ALL_TO_ALL,
    "all_to_allv": ALL_TO_ALL,
    "alltoall": ALL_TO_ALL,
    "broadcast": BROADCAST,
}

# Fallback: parse from NCCL kernel name (ncclDevKernel_AllGather_RING_LL...)
KERNEL_NAME_MAP = [
    ("AllReduce",     ALL_REDUCE),
    ("AllGather",     ALL_GATHER),
    ("ReduceScatter", REDUCE_SCATTER),
    ("AllToAll",      ALL_TO_ALL),
    ("Broadcast",     BROADCAST),
    ("SendRecv",      ALL_TO_ALL),
]


def infer_collective_type(kernel_name: str, coll_name: str) -> int | None:
    coll_lower = coll_name.strip().lower().replace(" ", "_")
    if coll_lower in COLL_NAME_MAP:
        return COLL_NAME_MAP[coll_lower]
    for kw, ctype in KERNEL_NAME_MAP:
        if kw.lower() in kernel_name.lower():
            return ctype
    return None


def dtype_to_bytes(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype, 2)  # default BFloat16


def comm_size_bytes(args: dict) -> int:
    dtype = args.get("dtype", "BFloat16")
    elem_bytes = dtype_to_bytes(dtype)
    nelems = args.get("In msg nelems", 0)
    if nelems == 0:
        nelems = args.get("Out msg nelems", 0)
    return int(nelems) * elem_bytes


def parse_pg_ranks(pg_ranks_str: str) -> list[int] | None:
    """Parse 'Process Group Ranks' field (a string like '[0, 1, 2, 3]') into a list."""
    if not pg_ranks_str:
        return None
    try:
        result = ast.literal_eval(pg_ranks_str)
        if isinstance(result, list):
            return sorted(int(r) for r in result)
    except Exception:
        pass
    return None


def load_trace_partial(path: Path, max_mb: int = 500) -> list:
    """Load traceEvents from a kineto JSON, with fallback partial-parse."""
    max_bytes = max_mb * 1024 * 1024
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        return data.get("traceEvents", [])
    except json.JSONDecodeError as e:
        cut = text.rfind("\n  },\n", 0, e.pos)
        if cut == -1:
            cut = text.rfind("},", 0, e.pos)
        text2 = text[:cut + 1] + "\n]}"
        te_idx = text2.find('"traceEvents"')
        if te_idx == -1:
            return []
        wrapped = '{"traceEvents":' + text2[te_idx + len('"traceEvents"'):]
        try:
            return json.loads(wrapped).get("traceEvents", [])
        except Exception:
            return []


def extract_nccl_events(events: list) -> list:
    """Return NCCL collective kernel events with msg size info, sorted by timestamp.

    SendRecv (P2P pipeline-parallel) events are excluded: they don't fit the flat
    topology model and cause deadlocks in AstraSim's collective simulation.
    """
    nccl = [
        e for e in events
        if e.get("cat") == "kernel"
        and ("ncclDev" in e.get("name", "") or "ncclKernel" in e.get("name", ""))
        and "SendRecv" not in e.get("name", "")
        and e.get("args", {}).get("In msg nelems", 0) > 0
    ]
    return sorted(nccl, key=lambda e: e.get("ts", 0))


def get_or_assign_group_id(
    pg_ranks: list[int],
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> int:
    """Return a stable unique integer ID for a process group identified by its rank list."""
    key = tuple(sorted(pg_ranks))
    if key not in group_registry:
        new_id = len(group_registry) + 1  # 1-indexed to avoid 0
        group_registry[key] = new_id
        group_npus[new_id] = list(key)
    return group_registry[key]


def generate_et(
    rank: int,
    nccl_events: list,
    output_dir: Path,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> int:
    """Write one Chakra ET file for the given rank. Returns number of collective nodes."""
    et_path = output_dir / f"chakra_trace.{rank}.et"
    node_id = 0
    nodes = []

    prev_end_ts = None
    for ev in nccl_events:
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        args = ev.get("args", {})
        coll_name = args.get("Collective name", "")
        kernel_name = ev.get("name", "")

        ctype = infer_collective_type(kernel_name, coll_name)
        if ctype is None:
            continue

        size_bytes = comm_size_bytes(args)
        if size_bytes == 0:
            continue

        # Resolve process group ID from the event's Process Group Ranks field.
        # ts/dur are in microseconds (Chrome trace format standard, regardless of
        # displayTimeUnit which only affects viewer display scaling).
        pg_ranks = parse_pg_ranks(args.get("Process Group Ranks", ""))
        if pg_ranks is not None:
            pg_id = get_or_assign_group_id(pg_ranks, group_registry, group_npus)
            pg_name_str = str(pg_id)
        else:
            pg_name_str = ""

        # Compute gap since previous collective ended
        if prev_end_ts is not None:
            gap_us = max(0.0, ts - prev_end_ts)
            if gap_us > 0:
                comp = ChakraNode()
                comp.id = node_id
                comp.name = f"compute_gap_{node_id}"
                comp.type = COMP_NODE
                comp.duration_micros = int(gap_us)
                if nodes:
                    comp.ctrl_deps.append(nodes[-1].id)
                comp.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                comp.attr.append(ChakraAttr(name="num_ops", int64_val=0))
                nodes.append(comp)
                node_id += 1

        comm = ChakraNode()
        comm.id = node_id
        comm.name = f"{coll_name or kernel_name.split('(')[0]}_{node_id}"
        comm.type = COMM_COLL_NODE
        if nodes:
            comm.ctrl_deps.append(nodes[-1].id)
        comm.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        comm.attr.append(ChakraAttr(name="comm_type", int64_val=ctype))
        comm.attr.append(ChakraAttr(name="comm_size", int64_val=size_bytes))
        if pg_name_str:
            comm.attr.append(ChakraAttr(name="pg_name", string_val=pg_name_str))
        nodes.append(comm)
        node_id += 1

        prev_end_ts = ts + dur

    with open(et_path, "wb") as f:
        encode_message(f, GlobalMetadata(version="0.0.4"))
        for node in nodes:
            encode_message(f, node)

    n_comm = sum(1 for n in nodes if n.type == COMM_COLL_NODE)
    print(f"  rank {rank}: {n_comm} collective nodes, "
          f"{len(nodes) - n_comm} compute gap nodes → {et_path.name}")
    return n_comm


def write_comm_group(output_dir: Path, group_npus: dict[int, list[int]]) -> bool:
    """Write comm_group.json. Returns True if any groups were written."""
    if not group_npus:
        return False
    comm_group = {str(gid): ranks for gid, ranks in sorted(group_npus.items())}
    (output_dir / "comm_group.json").write_text(json.dumps(comm_group, indent=2))
    print(f"[gen_chakra_et] comm_group.json: {len(comm_group)} unique process groups")
    for gid, ranks in sorted(group_npus.items()):
        print(f"  group {gid}: ranks {ranks}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", required=True, help="Comma-separated rank indices")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    ranks = [int(r) for r in args.ranks.split(",")]

    print(f"[gen_chakra_et] Generating Chakra ET files for {len(ranks)} ranks...")

    # Shared group registry across all ranks so group IDs are consistent.
    # group_registry: frozenset(ranks) → unique_id
    # group_npus:     unique_id → sorted list of NPU IDs
    group_registry: dict[tuple, int] = {}
    group_npus: dict[int, list[int]] = {}

    total_comm_nodes = 0
    for rank in ranks:
        trace_file = trace_dir / f"rank{rank}_trace.json"
        if not trace_file.exists():
            print(f"  WARNING: {trace_file} not found, skipping rank {rank}")
            continue
        events = load_trace_partial(trace_file)
        nccl_events = extract_nccl_events(events)
        if not nccl_events:
            print(f"  rank {rank}: no NCCL events found")
            continue
        total_comm_nodes += generate_et(
            rank, nccl_events, output_dir, group_registry, group_npus
        )

    if group_npus:
        write_comm_group(output_dir, group_npus)
    else:
        print("[gen_chakra_et] No process group info found; "
              "comm_group.json not written (AstraSim will use all-NPU default)")

    print(f"[gen_chakra_et] Done. Total collective nodes: {total_comm_nodes}")


if __name__ == "__main__":
    main()
