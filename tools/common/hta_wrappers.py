"""Thin wrappers over HTA analyzers to standardize outputs."""

from __future__ import annotations

from typing import Any, cast

from .torch_trace import load_hta_trace


def temporal_breakdown(trace_dir: str) -> dict[str, Any]:
    trace = load_hta_trace(trace_dir)
    if not hasattr(trace, "get_temporal_breakdown"):
        raise NotImplementedError("HTA version missing get_temporal_breakdown")
    df = trace.get_temporal_breakdown()
    if hasattr(df, "to_dict"):
        return cast("dict[str, Any]", df.to_dict(orient="index"))
    return cast("dict[str, Any]", df)


def comm_comp_overlap(trace_dir: str) -> dict[str, Any]:
    trace = load_hta_trace(trace_dir)
    if not hasattr(trace, "get_comm_comp_overlap"):
        raise NotImplementedError("HTA version missing get_comm_comp_overlap")
    df = trace.get_comm_comp_overlap()
    if hasattr(df, "to_dict"):
        return cast("dict[str, Any]", df.to_dict(orient="index"))
    return cast("dict[str, Any]", df)


def comm_stats(trace_dir: str) -> dict[str, Any]:
    trace = load_hta_trace(trace_dir)
    if not hasattr(trace, "get_comm_stats"):
        raise NotImplementedError("HTA version missing get_comm_stats")
    df = trace.get_comm_stats()
    if hasattr(df, "to_dict"):
        return cast("dict[str, Any]", df.to_dict(orient="index"))
    return cast("dict[str, Any]", df)


def kernel_breakdown(trace_dir: str) -> dict[str, Any]:
    trace = load_hta_trace(trace_dir)
    if not hasattr(trace, "get_kernel_breakdown"):
        raise NotImplementedError("HTA version missing get_kernel_breakdown")
    df = trace.get_kernel_breakdown()
    if hasattr(df, "to_dict"):
        return cast("dict[str, Any]", df.to_dict(orient="index"))
    return cast("dict[str, Any]", df)
