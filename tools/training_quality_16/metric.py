"""Training Quality metrics calculation.

Extracts training-related metrics from logs and traces:
- Training loss vs step
- Gradient norm
- Learning rate schedule

NVTX Dependency: None
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def metric_cal(directory: str, profile_mode: str = "auto") -> dict[str, Any]:
    """Extract training quality metrics from logs and traces.

    Args:
        directory: Path to the trace directory containing trace files and logs.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).

    Returns:
        Dictionary with training quality metrics:
            - loss_by_step: List of (step, loss) tuples
            - gradient_norm_by_step: List of (step, grad_norm) tuples
            - lr_by_step: List of (step, lr) tuples
            - has_loss: Whether loss data was found
            - has_grad_norm: Whether gradient norm data was found
            - has_lr: Whether learning rate data was found
    """
    trace_dir = Path(directory)

    result: dict[str, Any] = {
        "loss_by_step": [],
        "gradient_norm_by_step": [],
        "lr_by_step": [],
        "has_loss": False,
        "has_grad_norm": False,
        "has_lr": False,
    }

    # Look for log files
    log_files = list(trace_dir.glob("*.log"))
    log_files.extend(trace_dir.glob("*.out"))
    log_files.extend(trace_dir.glob("*.err"))

    # Also check parent directory for logs
    parent_dir = trace_dir.parent
    if parent_dir.exists():
        log_files.extend(parent_dir.glob("*.log"))
        log_files.extend(parent_dir.glob("*.out"))
        log_files.extend(parent_dir.glob("*.err"))

    # Parse logs for training metrics
    for log_file in log_files:
        if not log_file.is_file():
            continue

        try:
            loss_data = _extract_loss_from_log(log_file)
            if loss_data:
                result["loss_by_step"].extend(loss_data)
                result["has_loss"] = True

            grad_norm_data = _extract_grad_norm_from_log(log_file)
            if grad_norm_data:
                result["gradient_norm_by_step"].extend(grad_norm_data)
                result["has_grad_norm"] = True

            lr_data = _extract_lr_from_log(log_file)
            if lr_data:
                result["lr_by_step"].extend(lr_data)
                result["has_lr"] = True

        except Exception as e:
            print(f"Error parsing {log_file}: {e}", file=sys.stderr)
            continue

    # Also check trace files for embedded metrics
    trace_patterns = ["kineto_trace*.json", "rank*_trace.json", "*trace*.json"]
    trace_files: list[Path] = []
    for pattern in trace_patterns:
        trace_files.extend(trace_dir.glob(pattern))

    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.extend(profile_dir.glob(pattern))

    for trace_path in trace_files:
        if not trace_path.is_file() or trace_path.suffix != ".json":
            continue

        try:
            trace_metrics = _extract_metrics_from_trace(trace_path)
            if trace_metrics.get("loss"):
                result["loss_by_step"].extend(trace_metrics["loss"])
                result["has_loss"] = True
            if trace_metrics.get("grad_norm"):
                result["gradient_norm_by_step"].extend(trace_metrics["grad_norm"])
                result["has_grad_norm"] = True
            if trace_metrics.get("lr"):
                result["lr_by_step"].extend(trace_metrics["lr"])
                result["has_lr"] = True
        except Exception:
            continue

    # Sort by step
    result["loss_by_step"].sort(key=lambda x: x[0])
    result["gradient_norm_by_step"].sort(key=lambda x: x[0])
    result["lr_by_step"].sort(key=lambda x: x[0])

    # Print summary
    if result["has_loss"]:
        print(f"  Training Quality: {len(result['loss_by_step'])} loss values found", file=sys.stderr)
    if result["has_grad_norm"]:
        print(
            f"  Training Quality: {len(result['gradient_norm_by_step'])} grad_norm values found",
            file=sys.stderr,
        )
    if result["has_lr"]:
        print(f"  Training Quality: {len(result['lr_by_step'])} LR values found", file=sys.stderr)
    if not (result["has_loss"] or result["has_grad_norm"] or result["has_lr"]):
        print("  Training Quality: No training metrics found in logs or traces", file=sys.stderr)

    return result


def _extract_loss_from_log(log_file: Path) -> list[tuple[int, float]]:
    """Extract loss values from log file."""
    loss_data: list[tuple[int, float]] = []

    # Common loss patterns
    loss_patterns = [
        r"loss[:\s=]+([\d.]+)",
        r"train_loss[:\s=]+([\d.]+)",
        r"Loss[:\s=]+([\d.]+)",
        r"step[:\s]+(\d+).*loss[:\s=]+([\d.]+)",
        r"iter[:\s]+(\d+).*loss[:\s=]+([\d.]+)",
    ]

    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Try to extract step and loss
                for pattern in loss_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) == 1:
                            # Only loss, try to find step number in line
                            loss_val = float(groups[0])
                            step_match = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)
                            if step_match:
                                step = int(step_match.group(1))
                                loss_data.append((step, loss_val))
                        elif len(groups) == 2:
                            # Both step and loss
                            step = int(groups[0])
                            loss_val = float(groups[1])
                            loss_data.append((step, loss_val))
                        break
    except Exception:
        pass

    return loss_data


def _extract_grad_norm_from_log(log_file: Path) -> list[tuple[int, float]]:
    """Extract gradient norm from log file."""
    grad_norm_data: list[tuple[int, float]] = []

    grad_norm_patterns = [
        r"grad_norm[:\s=]+([\d.]+)",
        r"gradient_norm[:\s=]+([\d.]+)",
        r"grad[:\s]+norm[:\s=]+([\d.]+)",
        r"step[:\s]+(\d+).*grad_norm[:\s=]+([\d.]+)",
    ]

    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for pattern in grad_norm_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) == 1:
                            grad_norm = float(groups[0])
                            step_match = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)
                            if step_match:
                                step = int(step_match.group(1))
                                grad_norm_data.append((step, grad_norm))
                        elif len(groups) == 2:
                            step = int(groups[0])
                            grad_norm = float(groups[1])
                            grad_norm_data.append((step, grad_norm))
                        break
    except Exception:
        pass

    return grad_norm_data


def _extract_lr_from_log(log_file: Path) -> list[tuple[int, float]]:
    """Extract learning rate from log file."""
    lr_data: list[tuple[int, float]] = []

    lr_patterns = [
        r"lr[:\s=]+([\d.e-]+)",
        r"learning_rate[:\s=]+([\d.e-]+)",
        r"step[:\s]+(\d+).*lr[:\s=]+([\d.e-]+)",
    ]

    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for pattern in lr_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        if len(groups) == 1:
                            lr = float(groups[0])
                            step_match = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)
                            if step_match:
                                step = int(step_match.group(1))
                                lr_data.append((step, lr))
                        elif len(groups) == 2:
                            step = int(groups[0])
                            lr = float(groups[1])
                            lr_data.append((step, lr))
                        break
    except Exception:
        pass

    return lr_data


def _extract_metrics_from_trace(trace_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Extract training metrics from trace file metadata or events."""
    result: dict[str, list[tuple[int, float]]] = {
        "loss": [],
        "grad_norm": [],
        "lr": [],
    }

    try:
        with trace_path.open() as f:
            data = json.load(f)

        # Check metadata
        metadata = data.get("metadata", {})
        if "loss" in metadata:
            # Assume it's a dict mapping step -> loss
            if isinstance(metadata["loss"], dict):
                for step_str, loss_val in metadata["loss"].items():
                    try:
                        step = int(step_str)
                        loss = float(loss_val)
                        result["loss"].append((step, loss))
                    except (ValueError, TypeError):
                        pass

        # Check events for embedded metrics
        events = data.get("traceEvents", [])
        for event in events:
            args = event.get("args", {})

            # Look for loss/grad_norm/lr in event args
            if "loss" in args:
                step = event.get("ts", 0)  # Use timestamp as step proxy
                try:
                    loss = float(args["loss"])
                    result["loss"].append((int(step), loss))
                except (ValueError, TypeError):
                    pass

            if "grad_norm" in args or "gradient_norm" in args:
                step = event.get("ts", 0)
                try:
                    grad_norm = float(args.get("grad_norm") or args.get("gradient_norm"))
                    result["grad_norm"].append((int(step), grad_norm))
                except (ValueError, TypeError):
                    pass

            if "lr" in args or "learning_rate" in args:
                step = event.get("ts", 0)
                try:
                    lr = float(args.get("lr") or args.get("learning_rate"))
                    result["lr"].append((int(step), lr))
                except (ValueError, TypeError):
                    pass

    except Exception:
        pass

    return result

