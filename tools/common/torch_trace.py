"""Helpers for loading HTA-compatible traces."""

from __future__ import annotations

import importlib
from pathlib import Path


from hta.trace_analysis import TraceAnalysis


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


def load_hta_trace(trace_dir: str) -> TraceAnalysis:
    """Return an HTA TraceAnalysis object for the directory."""

    return TraceAnalysis(trace_dir)



