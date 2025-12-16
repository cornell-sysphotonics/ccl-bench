"""Utilities for loading run metadata required by multiple metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


_DEFAULT_METADATA_FILE = "run_metadata.json"


class MetadataError(RuntimeError):
    """Raised when required run metadata is missing or invalid."""


def load_run_metadata(trace_dir: str) -> dict[str, Any]:
    path = Path(trace_dir) / _DEFAULT_METADATA_FILE
    if not path.exists():
        raise MetadataError(
            f"Missing metadata file '{_DEFAULT_METADATA_FILE}' in {trace_dir}. Provide run metadata to compute throughput and MFU."
        )
    with path.open() as fh:
        return cast("dict[str, Any]", json.load(fh))


def tokens_per_step(meta: dict) -> int:
    if "tokens_per_step" not in meta:
        gbs = meta.get("global_batch_size")
        seq = meta.get("seq_len")
        if gbs is not None and seq is not None:
            return int(gbs) * int(seq)
        raise MetadataError("tokens_per_step missing from metadata")
    return int(meta["tokens_per_step"])


def hbm_bytes(meta: dict) -> int:
    if "gpu_hbm_bytes" not in meta:
        raise MetadataError("gpu_hbm_bytes missing from metadata")
    return int(meta["gpu_hbm_bytes"])


def flops_per_token(meta: dict) -> float:
    if "model_flops_per_token" not in meta:
        raise MetadataError("model_flops_per_token missing from metadata")
    return float(meta["model_flops_per_token"])


def peak_flops_per_gpu(meta: dict) -> float:
    if "gpu_peak_tflops" not in meta:
        raise MetadataError("gpu_peak_tflops missing from metadata")
    return float(meta["gpu_peak_tflops"])


def world_size(meta: dict) -> int:
    if "world_size" not in meta:
        raise MetadataError("world_size missing from metadata")
    return int(meta["world_size"])
