"""Helpers for loading HTA-compatible traces."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, TypeAlias, cast


class TraceAnalysisType(Protocol):
    """Interface for HTA TraceAnalysis used by these helpers."""

    def __init__(self, trace_dir: str) -> None:
        """Initialize the trace analysis for a given directory."""

    def get_temporal_breakdown(self) -> object:
        """Return temporal breakdown data."""

    def get_comm_comp_overlap(self) -> object:
        """Return communication/computation overlap data."""

    def get_comm_stats(self) -> object:
        """Return communication statistics."""

    def get_kernel_breakdown(self) -> object:
        """Return kernel breakdown data."""


TraceAnalysisClass: TypeAlias = type[TraceAnalysisType]
TraceAnalysis: TraceAnalysisClass | None = None

try:  # soft dependency
    _hta = importlib.import_module("hta.trace_analysis")
    TraceAnalysis = cast("TraceAnalysisClass", getattr(_hta, "TraceAnalysis", None))
except ModuleNotFoundError:
    TraceAnalysis = None


def list_ranks(trace_dir: str) -> list[int]:
    base = Path(trace_dir)
    ranks = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("rank"):
            try:
                ranks.append(int(child.name.replace("rank", "")))
            except ValueError:
                continue
    return sorted(ranks) if ranks else [0]


def load_hta_trace(trace_dir: str) -> TraceAnalysisType:
    """Return an HTA TraceAnalysis object for the directory.

    This is a lightweight wrapper that raises a helpful error if HTA is missing.
    """
    trace_cls = TraceAnalysis
    if trace_cls is None:
        raise ImportError(
            "Holistic Trace Analysis (hta) is required for torch profile parsing. Install with `pip install hta`."
        )

    assert trace_cls is not None  # for type checkers

    return trace_cls(trace_dir)
