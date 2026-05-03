#!/usr/bin/env python3
"""
Generate Chakra Execution Trace (.et) files from kineto-only rankN_trace.json files.

Runs inside the astra-sim Docker container where the chakra package is installed.
Extracts NCCL collective operations from each rank's trace and emits Chakra ET
nodes. The default compute model inserts measured rank-local gap COMP_NODEs
between globally ordered collectives. The kernels compute model emits non-NCCL
GPU kernels as replayed COMP_NODEs.

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
import multiprocessing
import os
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

_ET_WORKER_EVENTS_BY_RANK: dict[int, list] = {}
_ET_WORKER_TRACE_DIR: Path | None = None
_ET_WORKER_OUTPUT_DIR: Path | None = None
_ET_WORKER_GROUP_REGISTRY: dict[tuple, int] = {}
_ET_WORKER_GROUP_NPUS: dict[int, list[int]] = {}
_ET_WORKER_ANNOTATIONS_BY_RANK: dict[int, dict[tuple, dict]] = {}
_ET_WORKER_COMPUTE_MODEL = "gap"
_ET_WORKER_KERNEL_DEPENDENCY_MODE = "rank"

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


def is_gpu_kernel_event(event: dict) -> bool:
    return event.get("cat") == "kernel" and event.get("dur", 0) > 0


def is_nccl_kernel_name(name: str) -> bool:
    return "ncclDev" in name or "ncclKernel" in name


def has_message_size_metadata(args: dict) -> bool:
    return (
        int(args.get("In msg nelems", 0) or 0) > 0
        or int(args.get("Out msg nelems", 0) or 0) > 0
    )


def normalized_collective_name(args: dict) -> str:
    return args.get("Collective name", "").strip().lower().replace(" ", "_")


def is_tiny_sendrecv_control_event(event: dict) -> bool:
    args = event.get("args", {})
    return (
        "SendRecv" in event.get("name", "")
        and args.get("dtype") in {"Long", "Int64"}
        and int(args.get("In msg nelems", 0) or 0) <= 64
        and int(args.get("Out msg nelems", 0) or 0) <= 64
    )


def is_nccl_collective_event(event: dict) -> bool:
    if not (is_gpu_kernel_event(event) and is_nccl_kernel_name(event.get("name", ""))):
        return False
    args = event.get("args", {})
    # SendRecv carries all_to_allv payload transfers in MoE traces, but also tiny
    # control/barrier exchanges. Model only the payload collectives.
    if "SendRecv" in event.get("name", ""):
        if normalized_collective_name(args) != "all_to_allv":
            return False
        if is_tiny_sendrecv_control_event(event):
            return False
    return has_message_size_metadata(args)


def is_simulatable_nccl_collective_event(event: dict) -> bool:
    return is_nccl_collective_event(event) and not event.get("_skip_collective", False)


def stream_key(event: dict) -> str:
    args = event.get("args", {})
    for key in ("stream", "stream id", "stream_id", "cuda_stream"):
        if key in args:
            return f"stream:{args[key]}"
    return f"pid:{event.get('pid', '')}:tid:{event.get('tid', '')}"


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


def event_comm_size_bytes(event: dict) -> int:
    return int(event.get("_comm_size_override", comm_size_bytes(event.get("args", {}))))


def parse_pg_ranks(pg_ranks_str: str) -> list[int] | None:
    """Parse 'Process Group Ranks' field into a sorted int list.

    Handles both dense '[0, 1, 2, 3]' and Python ellipsis notation
    '[0, 1, ..., 31]' that torchtitan emits for large process groups.
    """
    if not pg_ranks_str:
        return None
    try:
        result = ast.literal_eval(pg_ranks_str)
        if isinstance(result, list):
            if any(item is ... for item in result):
                # Expand ellipsis: [a, b, ..., c] fills integers (b+1)..(c-1)
                expanded: list[int] = []
                for i, item in enumerate(result):
                    if item is ...:
                        if expanded and i + 1 < len(result):
                            for r in range(expanded[-1] + 1, int(result[i + 1])):
                                expanded.append(r)
                    else:
                        expanded.append(int(item))
                return sorted(expanded)
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


def nccl_event_key(event: dict) -> tuple:
    args = event.get("args", {})
    return (
        event.get("ts", 0),
        event.get("dur", 0),
        event.get("name", ""),
        event.get("pid", ""),
        event.get("tid", ""),
        args.get("External id", ""),
        args.get("Process Group Ranks", ""),
        args.get("Collective name", ""),
        args.get("dtype", ""),
        args.get("In msg nelems", ""),
        args.get("Out msg nelems", ""),
    )


def load_nccl_events_partial(path: Path) -> list:
    return [
        event for event in load_trace_partial(path)
        if is_nccl_collective_event(event)
    ]


def collect_nccl_annotations(events_by_rank: dict[int, list]) -> dict[int, dict[tuple, dict]]:
    annotations_by_rank: dict[int, dict[tuple, dict]] = {}
    annotation_keys = ("_global_comm_order", "_skip_collective", "_comm_size_override")
    for rank, events in events_by_rank.items():
        rank_annotations: dict[tuple, dict] = {}
        for event in events:
            annotations = {
                key: event[key]
                for key in annotation_keys
                if key in event
            }
            if annotations:
                rank_annotations[nccl_event_key(event)] = annotations
        annotations_by_rank[rank] = rank_annotations
    return annotations_by_rank


def apply_nccl_annotations(events: list, annotations: dict[tuple, dict]) -> None:
    if not annotations:
        return
    for event in events:
        if is_nccl_collective_event(event):
            event.update(annotations.get(nccl_event_key(event), {}))


def extract_nccl_events_by_pg(
    events: list,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> dict[int, list]:
    """Return NCCL collective events grouped by process group ID, sorted by timestamp.

    AstraSim pairs the N-th collective across all ranks in a group, so the ordering
    within each group must be globally consistent (timestamp order). Multiple CUDA
    streams that share the same process group are merged and sorted together.

    Independent process groups become independent ET chains — AstraSim issues their
    head nodes concurrently, matching real GPU stream parallelism.

    SendRecv all_to_allv payload kernels are included with per-instance sizes
    normalized across ranks. Tiny SendRecv control/barrier exchanges are excluded.
    """
    by_pg: dict[int, list] = {}
    for e in events:
        if not is_simulatable_nccl_collective_event(e):
            continue
        pg_ranks = parse_pg_ranks(e.get("args", {}).get("Process Group Ranks", ""))
        if pg_ranks is not None:
            pg_id = get_or_assign_group_id(pg_ranks, group_registry, group_npus)
        else:
            pg_id = 0  # unknown PG: group together in a fallback chain
        by_pg.setdefault(pg_id, []).append(e)
    for pg_id in by_pg:
        by_pg[pg_id].sort(key=lambda e: e.get("ts", 0))
    return by_pg


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


def make_compute_node(node_id: int, event: dict, name_prefix: str = "compute") -> ChakraNode:
    comp = ChakraNode()
    comp.id = node_id
    comp.name = f"{name_prefix}_{event.get('name', 'kernel')}_{node_id}"
    comp.type = COMP_NODE
    comp.duration_micros = max(1, int(event.get("dur", 0)))
    comp.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    comp.attr.append(ChakraAttr(name="num_ops", int64_val=0))
    return comp


def add_ctrl_dep(node: ChakraNode, dep_id: int | None) -> None:
    if dep_id is not None and dep_id not in node.ctrl_deps:
        node.ctrl_deps.append(dep_id)


def validate_ctrl_deps(nodes: list[ChakraNode], rank: int) -> None:
    node_ids = {node.id for node in nodes}
    for node in nodes:
        for dep_id in node.ctrl_deps:
            if dep_id not in node_ids:
                raise ValueError(
                    f"rank {rank}: node {node.id} has missing ctrl_dep {dep_id}"
                )


def make_comm_node(
    node_id: int,
    event: dict,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> tuple[ChakraNode | None, int | None]:
    args = event.get("args", {})
    coll_name = args.get("Collective name", "")
    kernel_name = event.get("name", "")

    ctype = infer_collective_type(kernel_name, coll_name)
    if ctype is None:
        return None, None

    size_bytes = event_comm_size_bytes(event)
    if size_bytes == 0:
        return None, None

    pg_id = None
    pg_name_str = ""
    pg_ranks = parse_pg_ranks(args.get("Process Group Ranks", ""))
    if pg_ranks is not None:
        pg_id = get_or_assign_group_id(pg_ranks, group_registry, group_npus)
        pg_name_str = str(pg_id)

    comm = ChakraNode()
    comm.id = node_id
    comm.name = f"{coll_name or kernel_name.split('(')[0]}_{node_id}"
    comm.type = COMM_COLL_NODE
    comm.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    comm.attr.append(ChakraAttr(name="comm_type", int64_val=ctype))
    comm.attr.append(ChakraAttr(name="comm_size", int64_val=size_bytes))
    if pg_name_str:
        comm.attr.append(ChakraAttr(name="pg_name", string_val=pg_name_str))
    return comm, pg_id


def _build_pg_chain(
    pg_events: list,
    node_id_start: int,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> tuple[list, int, dict[int, int]]:
    """Build a linear chain of COMP+COMM nodes for one process group's collectives.

    Returns (nodes, next_node_id, comm_nodes_by_order) where comm_nodes_by_order maps
    _global_comm_order → node_id for each comm node that carries the annotation.
    """
    nodes = []
    node_id = node_id_start
    prev_end_ts = None
    prev_node_id = None
    comm_nodes_by_order: dict[int, int] = {}

    for ev in pg_events:
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        # Insert a compute gap node for idle time since the last op on this stream.
        if prev_end_ts is not None:
            gap_us = max(0.0, ts - prev_end_ts)
            if gap_us > 0:
                comp = make_compute_node(
                    node_id, {"name": "gap", "dur": gap_us}, name_prefix="compute"
                )
                if prev_node_id is not None:
                    comp.ctrl_deps.append(prev_node_id)
                nodes.append(comp)
                prev_node_id = node_id
                node_id += 1

        comm, _pg_id = make_comm_node(node_id, ev, group_registry, group_npus)
        if comm is None:
            continue
        if prev_node_id is not None:
            comm.ctrl_deps.append(prev_node_id)
        nodes.append(comm)
        if "_global_comm_order" in ev:
            comm_nodes_by_order[int(ev["_global_comm_order"])] = node_id
        prev_node_id = node_id
        node_id += 1
        prev_end_ts = ts + dur

    return nodes, node_id, comm_nodes_by_order


def _build_ordered_gap_graph(
    events: list,
    node_id_start: int,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> tuple[list, int, int]:
    """Build one rank-local COMP+COMM chain sorted by canonical collective order.

    Older gap-mode ETs built one independent timeline per process group, then
    added cross-PG ordering dependencies. That can count the same wall-clock
    idle interval once per PG and create a large bandwidth-insensitive replay
    floor. This graph has a single gap timeline per rank, so measured gaps are
    replayed once while COMM_COLL_NODEs still carry their process-group IDs.
    """
    ordered_events = [
        event for event in events
        if is_simulatable_nccl_collective_event(event)
    ]
    ordered_events.sort(
        key=lambda event: (
            int(event.get("_global_comm_order", 10**18)),
            event.get("ts", 0),
            event.get("dur", 0),
            event.get("name", ""),
        )
    )

    nodes = []
    node_id = node_id_start
    prev_end_ts = None
    prev_node_id = None
    seen_pgs: set[int] = set()

    for event in ordered_events:
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if prev_end_ts is not None:
            gap_us = max(0.0, ts - prev_end_ts)
            if gap_us > 0:
                comp = make_compute_node(
                    node_id, {"name": "gap", "dur": gap_us}, name_prefix="compute"
                )
                add_ctrl_dep(comp, prev_node_id)
                nodes.append(comp)
                prev_node_id = node_id
                node_id += 1

        comm, pg_id = make_comm_node(node_id, event, group_registry, group_npus)
        if comm is None:
            continue
        add_ctrl_dep(comm, prev_node_id)
        nodes.append(comm)
        prev_node_id = node_id
        node_id += 1
        prev_end_ts = ts + dur
        if pg_id is not None:
            seen_pgs.add(pg_id)

    return nodes, node_id, len(seen_pgs)


def _build_kernel_graph(
    rank: int,
    events: list,
    node_id_start: int,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
    kernel_dependency_mode: str,
) -> tuple[list, int, int]:
    """Build replay COMP nodes for non-NCCL kernels plus NCCL COMM nodes.

    Dependencies always preserve CUDA stream order. In rank dependency mode,
    nodes also depend on the previously emitted kernel on the same rank, which
    constructs explicit compute↔communication ordering from trace timestamps.
    Positive idle intervals between emitted kernels are replayed as gap COMP_NODEs:
    rank mode inserts one rank-local gap chain, while stream mode inserts gaps
    on each CUDA stream independently.
    Collective nodes also depend on the previous collective in the same process
    group. ASTRA's analytical frontend has one GPU comm slot per rank; a
    rank-local comm order avoids deadlocks where different ranks hold different
    process-group collectives simultaneously.
    """
    nodes = []
    node_id = node_id_start
    prev_by_stream: dict[str, int] = {}
    prev_end_by_stream: dict[str, float] = {}
    prev_by_rank: int | None = None
    prev_rank_end_ts: float | None = None
    prev_rank_comm_order: int | None = None
    prev_comm_by_pg: dict[int, int] = {}
    comm_nodes_by_order: dict[int, ChakraNode] = {}
    n_comm = 0

    kernel_events = [
        e for e in events
        if is_gpu_kernel_event(e)
        and (
            not is_nccl_kernel_name(e.get("name", ""))
            or is_simulatable_nccl_collective_event(e)
        )
    ]
    kernel_events.sort(key=lambda e: (e.get("ts", 0), e.get("dur", 0), e.get("name", "")))

    def insert_gap(dep_id: int | None, gap_us: float) -> int | None:
        nonlocal node_id
        if dep_id is None or gap_us <= 0:
            return dep_id
        gap = make_compute_node(
            node_id, {"name": "gap", "dur": gap_us}, name_prefix="compute"
        )
        add_ctrl_dep(gap, dep_id)
        nodes.append(gap)
        gap_id = node_id
        node_id += 1
        return gap_id

    for ev in kernel_events:
        skey = stream_key(ev)
        name = ev.get("name", "")
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        stream_dep = prev_by_stream.get(skey)
        rank_dep = prev_by_rank

        if is_simulatable_nccl_collective_event(ev):
            comm_order = (
                int(ev["_global_comm_order"])
                if "_global_comm_order" in ev else None
            )
            rank_dep_allowed = (
                kernel_dependency_mode == "rank"
                and (
                    comm_order is None
                    or prev_rank_comm_order is None
                    or prev_rank_comm_order <= comm_order
                )
            )
            if kernel_dependency_mode == "stream":
                stream_dep = insert_gap(
                    stream_dep, ts - prev_end_by_stream.get(skey, ts)
                )
            elif rank_dep_allowed:
                rank_dep = insert_gap(
                    rank_dep, ts - prev_rank_end_ts
                    if prev_rank_end_ts is not None else 0
                )

            node, pg_id = make_comm_node(node_id, ev, group_registry, group_npus)
            if node is None:
                continue
            add_ctrl_dep(node, stream_dep)
            if rank_dep_allowed:
                add_ctrl_dep(node, rank_dep)
            if pg_id is not None and pg_id in prev_comm_by_pg:
                add_ctrl_dep(node, prev_comm_by_pg[pg_id])
            nodes.append(node)
            current_node_id = node.id
            prev_by_stream[skey] = current_node_id
            prev_by_rank = current_node_id
            prev_end_by_stream[skey] = max(
                prev_end_by_stream.get(skey, ts), ts + dur
            )
            prev_rank_end_ts = max(
                prev_rank_end_ts if prev_rank_end_ts is not None else ts,
                ts + dur,
            )
            prev_rank_comm_order = comm_order
            if pg_id is not None:
                prev_comm_by_pg[pg_id] = current_node_id
            if "_global_comm_order" in ev:
                comm_nodes_by_order[int(ev["_global_comm_order"])] = node
            node_id += 1
            n_comm += 1
        elif not is_nccl_kernel_name(name):
            if kernel_dependency_mode == "stream":
                stream_dep = insert_gap(
                    stream_dep, ts - prev_end_by_stream.get(skey, ts)
                )
            elif kernel_dependency_mode == "rank":
                rank_dep = insert_gap(
                    rank_dep, ts - prev_rank_end_ts
                    if prev_rank_end_ts is not None else 0
                )

            node = make_compute_node(node_id, ev, name_prefix="kernel")
            add_ctrl_dep(node, stream_dep)
            if kernel_dependency_mode == "rank":
                add_ctrl_dep(node, rank_dep)
            nodes.append(node)
            current_node_id = node.id
            prev_by_stream[skey] = current_node_id
            prev_by_rank = current_node_id
            prev_end_by_stream[skey] = max(
                prev_end_by_stream.get(skey, ts), ts + dur
            )
            prev_rank_end_ts = max(
                prev_rank_end_ts if prev_rank_end_ts is not None else ts,
                ts + dur,
            )
            node_id += 1

    prev_order_node_id = None
    for order in sorted(comm_nodes_by_order):
        node = comm_nodes_by_order[order]
        add_ctrl_dep(node, prev_order_node_id)
        prev_order_node_id = node.id

    return nodes, node_id, n_comm


def annotate_global_comm_order(
    events_by_rank: dict[int, list],
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
) -> None:
    """Annotate NCCL events with a canonical cross-rank collective order.

    Ranks can observe independent process groups in different timestamp order.
    ASTRA has one GPU comm slot per rank, so entering those groups in inconsistent
    order can deadlock. We assign each collective instance a global order using
    (process group, instance index within that group), then sort those instances
    by their earliest observed timestamp.
    """
    occurrences: dict[tuple, list[dict]] = {}

    for rank in sorted(events_by_rank):
        by_pg: dict[int, list] = {}
        for event in events_by_rank[rank]:
            if not is_nccl_collective_event(event):
                continue
            event["_trace_rank"] = rank
            pg_ranks = parse_pg_ranks(
                event.get("args", {}).get("Process Group Ranks", "")
            )
            if pg_ranks is not None:
                pg_id = get_or_assign_group_id(pg_ranks, group_registry, group_npus)
            else:
                pg_id = 0
            by_pg.setdefault(pg_id, []).append(event)

        for pg_id, pg_events in by_pg.items():
            pg_events.sort(key=lambda e: (e.get("ts", 0), e.get("dur", 0), e.get("name", "")))
            for index, event in enumerate(pg_events):
                args = event.get("args", {})
                if "External id" in args:
                    key = (pg_id, "external", args["External id"])
                else:
                    key = (pg_id, "ordinal", index)
                occurrences.setdefault(key, []).append(event)

    skipped = 0
    all_ranks = sorted(events_by_rank)
    for key, events in occurrences.items():
        pg_id = int(key[0])
        expected_ranks = group_npus.get(pg_id, all_ranks)
        present_ranks = sorted({int(event.get("_trace_rank", -1)) for event in events})
        ctypes = {
            infer_collective_type(
                event.get("name", ""),
                event.get("args", {}).get("Collective name", ""),
            )
            for event in events
        }
        coll_names = {normalized_collective_name(event.get("args", {})) for event in events}
        dtypes = {event.get("args", {}).get("dtype", "") for event in events}
        sizes = {comm_size_bytes(event.get("args", {})) for event in events}
        if (
            present_ranks != expected_ranks
            or len(ctypes) != 1
            or len(coll_names) != 1
            or len(dtypes) != 1
            or ("all_to_allv" not in coll_names and len(sizes) != 1)
        ):
            for event in events:
                event["_skip_collective"] = True
            skipped += len(events)
        elif "all_to_allv" in coll_names:
            # AstraSim's ALL_TO_ALL node expects the same comm_size on every rank.
            # all_to_allv traces carry per-rank byte counts, so use the largest
            # observed payload for this logical collective as a conservative model.
            size_bytes = max(comm_size_bytes(event.get("args", {})) for event in events)
            for event in events:
                event["_comm_size_override"] = size_bytes

    if skipped:
        print(
            "[gen_chakra_et] Skipping "
            f"{skipped} incomplete/inconsistent NCCL collective event(s)"
        )

    ordered_keys = sorted(
        [
            key for key, events in occurrences.items()
            if not any(event.get("_skip_collective", False) for event in events)
        ],
        key=lambda key: (
            min(event.get("ts", 0) for event in occurrences[key]),
            key[0],
            str(key[1]),
            key[2],
        ),
    )
    for order, key in enumerate(ordered_keys):
        for event in occurrences[key]:
            event["_global_comm_order"] = order


def generate_et(
    rank: int,
    events: list,
    output_dir: Path,
    group_registry: dict[tuple, int],
    group_npus: dict[int, list[int]],
    compute_model: str,
    kernel_dependency_mode: str,
) -> int:
    """Write one Chakra ET file for the given rank. Returns number of collective nodes.

    In gap mode, collectives become one rank-local ordered chain. Compute gaps
    capture idle time between consecutive NCCL collectives once per rank, and
    COMM_COLL_NODEs carry process-group IDs for AstraSim. The legacy pg-gap mode
    keeps the older independent-PG-chain behavior for comparison.
    """
    et_path = output_dir / f"chakra_trace.{rank}.et"
    all_nodes = []
    node_id = 0

    if compute_model == "gap":
        all_nodes, node_id, n_pgs = _build_ordered_gap_graph(
            events, node_id, group_registry, group_npus
        )
    elif compute_model == "pg-gap":
        events_by_pg = extract_nccl_events_by_pg(events, group_registry, group_npus)
        all_comm_nodes_by_order: dict[int, int] = {}
        for pg_id in sorted(events_by_pg):
            chain, node_id, chain_order = _build_pg_chain(
                events_by_pg[pg_id], node_id, group_registry, group_npus
            )
            all_nodes.extend(chain)
            all_comm_nodes_by_order.update(chain_order)
        n_pgs = len(events_by_pg)

        # Add cross-chain ctrl_deps to enforce global collective ordering.
        # AstraSim has one comm slot per rank (LIFO); without this, independent
        # chains can enter different collectives in different orders across ranks,
        # deadlocking on collectives that require all-rank synchronization.
        nodes_by_id = {n.id: n for n in all_nodes}
        prev_order_node_id = None
        for order in sorted(all_comm_nodes_by_order):
            nid = all_comm_nodes_by_order[order]
            node = nodes_by_id[nid]
            if prev_order_node_id is not None and prev_order_node_id not in node.ctrl_deps:
                node.ctrl_deps.append(prev_order_node_id)
            prev_order_node_id = nid
    elif compute_model == "kernels":
        all_nodes, node_id, _n_comm = _build_kernel_graph(
            rank, events, node_id, group_registry, group_npus,
            kernel_dependency_mode
        )
        n_pgs = len(group_npus)
    else:
        raise ValueError(f"unknown compute model: {compute_model}")

    validate_ctrl_deps(all_nodes, rank)

    with open(et_path, "wb") as f:
        encode_message(f, GlobalMetadata(version="0.0.4"))
        for node in all_nodes:
            encode_message(f, node)

    n_comm = sum(1 for n in all_nodes if n.type == COMM_COLL_NODE)
    n_comp = len(all_nodes) - n_comm
    comp_label = (
        "compute gap"
        if compute_model in {"gap", "pg-gap"}
        else "kernel compute"
    )
    graph_label = (
        f"{n_pgs} process group(s), ordered rank gap chain"
        if compute_model == "gap"
        else f"{n_pgs} independent PG chain(s)"
    )
    print(f"  rank {rank}: {n_comm} collective nodes, "
          f"{n_comp} {comp_label} nodes, "
          f"{graph_label} → {et_path.name}")
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


def generate_et_worker(rank: int) -> int:
    if _ET_WORKER_TRACE_DIR is None:
        raise RuntimeError("ET worker trace directory was not initialized")
    if _ET_WORKER_OUTPUT_DIR is None:
        raise RuntimeError("ET worker output directory was not initialized")
    events = load_trace_partial(_ET_WORKER_TRACE_DIR / f"rank{rank}_trace.json")
    apply_nccl_annotations(events, _ET_WORKER_ANNOTATIONS_BY_RANK.get(rank, {}))
    return generate_et(
        rank,
        events,
        _ET_WORKER_OUTPUT_DIR,
        dict(_ET_WORKER_GROUP_REGISTRY),
        {gid: list(npus) for gid, npus in _ET_WORKER_GROUP_NPUS.items()},
        _ET_WORKER_COMPUTE_MODEL,
        _ET_WORKER_KERNEL_DEPENDENCY_MODE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", required=True, help="Comma-separated rank indices")
    parser.add_argument("--compute-model", choices=["gap", "pg-gap", "kernels"], default="gap",
                        help="gap: insert measured rank-local gaps between globally "
                             "ordered collectives; pg-gap: legacy per-process-group "
                             "gap chains; "
                             "kernels: replay non-NCCL GPU kernels as COMP_NODEs")
    parser.add_argument("--kernel-dependency-mode", choices=["rank", "stream"], default="rank",
                        help="kernels mode only: rank serializes emitted compute and "
                             "communication kernels by rank-local timestamp; stream "
                             "preserves only CUDA stream ordering")
    parser.add_argument("--et-workers", type=int, default=None,
                        help="parallel workers for Chakra ET file generation "
                             "(default: min(8, ranks, CPU count))")
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

    nccl_events_by_rank: dict[int, list] = {}
    for rank in ranks:
        trace_file = trace_dir / f"rank{rank}_trace.json"
        if not trace_file.exists():
            print(f"  WARNING: {trace_file} not found, skipping rank {rank}")
            continue
        nccl_events_by_rank[rank] = load_nccl_events_partial(trace_file)

    annotate_global_comm_order(nccl_events_by_rank, group_registry, group_npus)
    annotations_by_rank = collect_nccl_annotations(nccl_events_by_rank)

    et_ranks = []
    for rank in sorted(nccl_events_by_rank):
        events = nccl_events_by_rank[rank]
        if not any(is_simulatable_nccl_collective_event(e) for e in events):
            print(f"  rank {rank}: no NCCL events found")
            continue
        et_ranks.append(rank)

    total_comm_nodes = 0
    max_workers = args.et_workers
    if max_workers is None:
        max_workers = min(8, len(et_ranks), os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, len(et_ranks) or 1))
    print(f"[gen_chakra_et] Writing Chakra ET files with {max_workers} worker(s)")

    if max_workers == 1:
        for rank in et_ranks:
            events = load_trace_partial(trace_dir / f"rank{rank}_trace.json")
            apply_nccl_annotations(events, annotations_by_rank.get(rank, {}))
            total_comm_nodes += generate_et(
                rank,
                events,
                output_dir,
                dict(group_registry),
                {gid: list(npus) for gid, npus in group_npus.items()},
                args.compute_model,
                args.kernel_dependency_mode,
            )
    else:
        global _ET_WORKER_EVENTS_BY_RANK
        global _ET_WORKER_TRACE_DIR
        global _ET_WORKER_OUTPUT_DIR
        global _ET_WORKER_GROUP_REGISTRY
        global _ET_WORKER_GROUP_NPUS
        global _ET_WORKER_ANNOTATIONS_BY_RANK
        global _ET_WORKER_COMPUTE_MODEL
        global _ET_WORKER_KERNEL_DEPENDENCY_MODE

        _ET_WORKER_EVENTS_BY_RANK = nccl_events_by_rank
        _ET_WORKER_TRACE_DIR = trace_dir
        _ET_WORKER_OUTPUT_DIR = output_dir
        _ET_WORKER_GROUP_REGISTRY = group_registry
        _ET_WORKER_GROUP_NPUS = group_npus
        _ET_WORKER_ANNOTATIONS_BY_RANK = annotations_by_rank
        _ET_WORKER_COMPUTE_MODEL = args.compute_model
        _ET_WORKER_KERNEL_DEPENDENCY_MODE = args.kernel_dependency_mode

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=max_workers) as pool:
            for n_comm in pool.imap_unordered(generate_et_worker, et_ranks):
                total_comm_nodes += n_comm

    if group_npus:
        write_comm_group(output_dir, group_npus)
    else:
        print("[gen_chakra_et] No process group info found; "
              "comm_group.json not written (AstraSim will use all-NPU default)")

    print(f"[gen_chakra_et] Done. Total collective nodes: {total_comm_nodes}")


if __name__ == "__main__":
    main()
