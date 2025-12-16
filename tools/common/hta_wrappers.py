"""Thin wrappers over HTA analyzers to standardize outputs."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING, Any, cast
import warnings

from hta.trace_analysis import TraceAnalysis


if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _suppress_hta_warnings() -> Iterator[None]:
    """Temporarily silence HTA warnings (e.g., single-iteration notice)."""
    hta_logger = logging.getLogger("hta")
    prev_level = hta_logger.level
    hta_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hta_logger.setLevel(prev_level)


def temporal_breakdown(trace_dir: str) -> dict[str, Any]:
    with _suppress_hta_warnings():
        trace = TraceAnalysis(trace_dir=trace_dir)
    if not hasattr(trace, "get_temporal_breakdown"):
        raise NotImplementedError("HTA version missing get_temporal_breakdown")
    df = trace.get_temporal_breakdown()
    if hasattr(df, "to_dict"):
        return cast("dict[str, Any]", df.to_dict(orient="index"))
    return cast("dict[str, Any]", df)


def comm_comp_overlap(trace_dir: str) -> dict[str, Any]:
    with _suppress_hta_warnings(), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        trace = TraceAnalysis(trace_dir=trace_dir)
    if not hasattr(trace, "get_comm_comp_overlap"):
        raise NotImplementedError("HTA version missing get_comm_comp_overlap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = trace.get_comm_comp_overlap(visualize=False)
    if hasattr(df, "fillna"):
        df = df.fillna(0)
    if hasattr(df, "to_dict"):
        return cast("dict[str, Any]", df.to_dict(orient="index"))
    return cast("dict[str, Any]", df)


def kernel_breakdown(trace_dir: str) -> dict[str, Any]:
    with _suppress_hta_warnings():
        trace = TraceAnalysis(trace_dir=trace_dir)
    if not hasattr(trace, "get_gpu_kernel_breakdown"):
        raise NotImplementedError("HTA version missing get_gpu_kernel_breakdown")
    breakdown: Any = trace.get_gpu_kernel_breakdown(visualize=False)
    # HTA returns Tuple[DataFrame, DataFrame]; convert both
    try:
        df1, df2 = breakdown
        return {
            "by_type": df1.to_dict(orient="index") if hasattr(df1, "to_dict") else df1,
            "by_kernel": df2.to_dict(orient="index") if hasattr(df2, "to_dict") else df2,
        }
    except Exception:
        if hasattr(breakdown, "to_dict"):
            return cast("dict[str, Any]", breakdown.to_dict(orient="index"))
        return cast("dict[str, Any]", breakdown)
