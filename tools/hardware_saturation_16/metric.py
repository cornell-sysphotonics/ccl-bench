"""Hardware Saturation metric calculation.

Measures GPU utilization metrics:
- MFU (Model FLOP Utilization) / TFLOPs per GPU
- Peak GPU memory (HBM) per rank

NVTX Dependency: None (uses kernel events and memory events)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Try to import tomli (for Python < 3.11) or tomllib (for Python >= 3.11)
try:
    import tomllib

    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[import-not-found]

        HAS_TOML = True
    except ImportError:
        HAS_TOML = False
        tomllib = None  # type: ignore[assignment]


def metric_cal(
    directory: str,
    profile_mode: str = "auto",
    workload_card_path: str | None = None,
) -> dict[str, Any]:
    """Calculate hardware saturation metrics from trace data.

    Args:
        directory: Path to the trace directory containing trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).
        workload_card_path: Optional explicit path to workload card YAML or TOML file.

    Returns:
        Dictionary with hardware saturation metrics:
            - mfu_percent: Model FLOP Utilization percentage
            - tflops_per_gpu: TFLOPs per second per GPU
            - peak_memory_gb: Dict mapping rank -> peak memory in GB
            - avg_memory_gb: Dict mapping rank -> average memory in GB
            - memory_by_rank: List of (rank, peak_gb) tuples sorted by rank
            - num_gpus: Number of GPUs detected
            - gpu_model: GPU model if detected
    """
    trace_dir = Path(directory)

    # Load workload card for model info - use explicit path if provided
    if workload_card_path:
        workload_card = _load_workload_card_from_path(Path(workload_card_path))
    else:
        workload_card = _load_workload_card(directory)

    # Extract memory usage per rank
    memory_data = _extract_memory_usage(trace_dir)

    # If no memory found in traces, try to estimate from model if we have workload card
    if not memory_data.get("memory_by_rank") and workload_card:
        estimated_memory = _estimate_memory_from_model(workload_card)
        if estimated_memory > 0:
            # Create a placeholder entry for rank 0
            memory_data["peak_memory_gb"] = {0: estimated_memory}
            memory_data["avg_memory_gb"] = {0: estimated_memory}
            memory_data["memory_by_rank"] = [(0, estimated_memory)]
            print(
                f"  Note: Using estimated memory from model: {estimated_memory:.1f} GB "
                "(memory profiling not enabled in traces)",
                file=sys.stderr,
            )

    # Calculate MFU/TFLOPs if we have model info
    mfu_data: dict[str, Any] = {}
    if workload_card:
        mfu_data = _calculate_mfu(trace_dir, workload_card)
        if not mfu_data or mfu_data.get("mfu_percent", 0) == 0:
            print(
                "  Warning: MFU calculation returned 0 or failed. "
                "Check workload card and iteration time.",
                file=sys.stderr,
            )
    else:
        print(
            "  Warning: No workload card found. MFU/TFLOPs cannot be calculated.",
            file=sys.stderr,
        )
        mfu_data = {
            "mfu_percent": 0.0,
            "tflops_per_gpu": 0.0,
            "note": "No workload card found",
        }

    result = {
        "peak_memory_gb": memory_data.get("peak_memory_gb", {}),
        "avg_memory_gb": memory_data.get("avg_memory_gb", {}),
        "memory_by_rank": memory_data.get("memory_by_rank", []),
        "num_gpus": memory_data.get("num_gpus", 0),
        "gpu_model": memory_data.get("gpu_model", "unknown"),
    }

    result.update(mfu_data)

    # Print summary for debugging
    if result.get("mfu_percent", 0) > 0:
        print(
            f"  Hardware Saturation: MFU={result['mfu_percent']:.1f}%, "
            f"TFLOPs={result.get('tflops_per_gpu', 0):.1f}",
            file=sys.stderr,
        )
    if result.get("memory_by_rank"):
        print(f"  Memory: {len(result['memory_by_rank'])} ranks detected", file=sys.stderr)
    else:
        print("  Warning: No memory data found in traces", file=sys.stderr)

    return result


def _load_workload_card_from_path(card_path: Path) -> dict[str, Any] | None:
    """Load workload card from an explicit file path."""
    if not card_path.exists():
        print(f"Warning: Workload card path does not exist: {card_path}", file=sys.stderr)
        return None

    try:
        if card_path.suffix.lower() in [".yaml", ".yml"]:
            with card_path.open() as f:
                return yaml.safe_load(f)
        elif card_path.suffix.lower() == ".toml":
            if HAS_TOML and tomllib is not None:
                with card_path.open("rb") as f:
                    data = tomllib.load(f)
                    if data:
                        # Convert TOML structure to workload card format
                        converted = _convert_toml_to_workload_card(data)
                        if converted:
                            print(f"Loaded TOML config: {card_path}", file=sys.stderr)
                            return converted
            else:
                print(
                    f"Warning: tomli not available. Cannot read TOML file: {card_path}",
                    file=sys.stderr,
                )
                print("         Install with: pip install tomli", file=sys.stderr)
    except Exception as e:
        print(f"Error loading {card_path}: {e}", file=sys.stderr)

    return None


def _convert_toml_to_workload_card(toml_data: dict[str, Any]) -> dict[str, Any] | None:
    """Convert TOML config structure to workload card YAML structure."""
    result: dict[str, Any] = {
        "parallelism": {},
        "workload": {
            "data": {},
            "model": {},
        },
        "hardware": {},
    }

    # Extract parallelism config
    parallel = toml_data.get("parallelism", {})
    if parallel:
        dp_shard = parallel.get("data_parallel_shard_degree", 1)
        dp_repl = parallel.get("data_parallel_replicate_degree", 1)
        result["parallelism"] = {
            "world_size": 0,  # Will be calculated
            "tp": parallel.get("tensor_parallel_degree", 1),
            "dp": dp_shard * dp_repl,
            "pp": parallel.get("pipeline_parallel_degree", 1),
            "cp": parallel.get("context_parallel_degree", 1),
            "ep": parallel.get("expert_parallel_degree", 1),
            "dp_shard": dp_shard,
            "dp_repl": dp_repl,
        }
        # Calculate world_size
        result["parallelism"]["world_size"] = (
            result["parallelism"]["dp"]
            * result["parallelism"]["tp"]
            * result["parallelism"]["pp"]
            * result["parallelism"]["cp"]
            * result["parallelism"]["ep"]
        )

    # Extract data config
    training = toml_data.get("training", {})
    if training:
        result["workload"]["data"] = {
            "seq_len": training.get("seq_len", 0),
            "batch_size": training.get("local_batch_size", 0),
            "global_batch_size": training.get("global_batch_size", 0),
            "local_batch_size": training.get("local_batch_size", 0),
            "micro_batch_size": training.get("pipeline_parallel_microbatch_size", 0),
        }

    # Extract model config
    model = toml_data.get("model", {})
    if model:
        model_name = model.get("name", "")
        flavor = model.get("flavor", "")
        full_name = f"{model_name}{flavor}" if flavor else model_name

        result["workload"]["model"] = {
            "name": full_name,
            "num_params": 0,  # Would need to look up from model registry
            "hidden_size": 0,
            "num_layers": 0,
            "num_heads": 0,
        }

        # Try to estimate params from name
        if "8b" in full_name.lower():
            result["workload"]["model"]["num_params"] = 8e9
        elif "70b" in full_name.lower():
            result["workload"]["model"]["num_params"] = 70e9
        elif "13b" in full_name.lower():
            result["workload"]["model"]["num_params"] = 13e9

    # Extract hardware info (if available in TOML)
    hardware = toml_data.get("hardware", {})
    if hardware:
        result["hardware"] = {
            "gpu_model": hardware.get("gpu_model", "unknown"),
            "num_gpus": hardware.get("num_gpus", result["parallelism"].get("world_size", 0)),
            "interconnect": hardware.get("interconnect", "unknown"),
            "gpu_memory_gb": hardware.get("gpu_memory_gb", 0),
        }
    else:
        # Default hardware info
        result["hardware"] = {
            "gpu_model": "unknown",
            "num_gpus": result["parallelism"].get("world_size", 0),
            "interconnect": "unknown",
            "gpu_memory_gb": 0,
        }

    return result if result.get("parallelism") or result.get("workload") else None


def _load_workload_card(directory: str) -> dict[str, Any] | None:
    """Load workload card YAML from directory or parent directories."""
    possible_names = [
        "workload_card.yaml",
        "workload_card_tp.yaml",
        "workload_card_pp.yaml",
        "workload_card_dp_tp.yaml",
        "workload_card_dp_pp.yaml",
        "workload_card_3d.yaml",
    ]

    # Check current directory
    for name in possible_names:
        card_path = Path(directory) / name
        if card_path.exists():
            try:
                with card_path.open() as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading {card_path}: {e}", file=sys.stderr)

    # Check parent directory (for iteration subdirectories)
    parent_dir = Path(directory).parent
    for name in possible_names:
        card_path = parent_dir / name
        if card_path.exists():
            try:
                with card_path.open() as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading {card_path}: {e}", file=sys.stderr)

    # Check grandparent directory (trace_base)
    grandparent_dir = parent_dir.parent
    for name in possible_names:
        card_path = grandparent_dir / name
        if card_path.exists():
            try:
                with card_path.open() as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading {card_path}: {e}", file=sys.stderr)

    return None


def _extract_memory_usage(trace_dir: Path) -> dict[str, Any]:
    """Extract GPU memory usage from trace files."""
    peak_memory_gb: dict[int, float] = {}
    avg_memory_gb: dict[int, float] = {}
    memory_samples: dict[int, list[float]] = {}
    gpu_model = "unknown"

    # Find trace files
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: list[Path] = []
    for pattern in trace_patterns:
        trace_files.extend(trace_dir.glob(pattern))

    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.extend(profile_dir.glob(pattern))

    # Deduplicate
    trace_files = list(set(trace_files))

    for trace_path in trace_files:
        if not trace_path.is_file() or trace_path.suffix != ".json":
            continue

        try:
            with trace_path.open() as f:
                data = json.load(f)

            events = data.get("traceEvents", [])

            # Extract rank from filename or metadata
            rank = _extract_rank_from_filename(trace_path)

            # Look for memory allocation events
            for event in events:
                cat = event.get("cat", "")
                name = event.get("name", "")
                args = event.get("args", {})

                # Look for memory events - check multiple patterns
                is_memory_event = (
                    cat == "memory"
                    or (cat == "user_annotation" and ("memory" in name.lower() or "Memory" in name))
                    or "memory" in name.lower()
                    or "Memory" in name
                    or "alloc" in name.lower()
                    or "reserve" in name.lower()
                    or name == "Memory Snapshot"
                    or name == "GPU Memory"
                )

                if is_memory_event:
                    # Try to extract memory size from various possible fields
                    size_bytes = None

                    # Check various possible fields in args
                    memory_field_names = [
                        "size",
                        "bytes",
                        "allocated_bytes",
                        "reserved_bytes",
                        "Allocated Bytes",
                        "Reserved Bytes",
                        "current_allocated",
                        "current_reserved",
                        "Size",
                        "Bytes",
                        "allocated",
                        "reserved",
                        "Device Total Allocated",
                        "Device Total Reserved",
                        "total_allocated",
                        "total_reserved",
                    ]
                    for key in memory_field_names:
                        if key in args:
                            val = args[key]
                            if isinstance(val, (int, float)) and val > 0:
                                size_bytes = val
                                break

                    # Also check nested structures in args
                    if size_bytes is None:
                        for nested_key in ["snapshot", "memory", "Memory", "allocated", "reserved"]:
                            if nested_key in args and isinstance(args[nested_key], dict):
                                nested = args[nested_key]
                                for key in memory_field_names[:8]:
                                    if key in nested:
                                        val = nested[key]
                                        if isinstance(val, (int, float)) and val > 0:
                                            size_bytes = val
                                            break
                                if size_bytes:
                                    break

                    if size_bytes and isinstance(size_bytes, (int, float)) and size_bytes > 0:
                        memory_gb = size_bytes / (1024**3)
                        if rank not in memory_samples:
                            memory_samples[rank] = []
                        memory_samples[rank].append(memory_gb)

                # Also look for memory info in any event args
                if not is_memory_event:
                    for key in [
                        "Device Total Allocated",
                        "Device Total Reserved",
                        "total_allocated_bytes",
                        "total_reserved_bytes",
                    ]:
                        if key in args:
                            val = args[key]
                            if isinstance(val, (int, float)) and val > 0:
                                memory_gb = val / (1024**3)
                                if rank not in memory_samples:
                                    memory_samples[rank] = []
                                memory_samples[rank].append(memory_gb)
                                break

                # Also check for GPU model info
                if "gpu_model" in args or "device_name" in args:
                    model = args.get("gpu_model") or args.get("device_name", "")
                    if model and gpu_model == "unknown":
                        gpu_model = str(model)

            # Also try to find memory info in metadata
            metadata = data.get("metadata", {})
            if "gpu_model" in metadata:
                gpu_model = metadata["gpu_model"]

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error reading {trace_path}: {e}", file=sys.stderr)
            continue

    # Calculate peak and average per rank
    for rank, samples in memory_samples.items():
        if samples:
            peak_memory_gb[rank] = max(samples)
            avg_memory_gb[rank] = sum(samples) / len(samples)

    # If no memory events found, try alternative method: look for memory snapshots
    if not peak_memory_gb:
        peak_memory_gb, avg_memory_gb = _extract_memory_from_snapshots(trace_files)

    # Create sorted list of (rank, peak_gb) tuples
    memory_by_rank = sorted(peak_memory_gb.items())

    return {
        "peak_memory_gb": peak_memory_gb,
        "avg_memory_gb": avg_memory_gb,
        "memory_by_rank": memory_by_rank,
        "num_gpus": len(peak_memory_gb) if peak_memory_gb else len(trace_files),
        "gpu_model": gpu_model,
    }


def _extract_memory_from_snapshots(
    trace_files: list[Path],
) -> tuple[dict[int, float], dict[int, float]]:
    """Try to extract memory from memory snapshot events."""
    peak_memory_gb: dict[int, float] = {}
    avg_memory_gb: dict[int, float] = {}
    memory_samples: dict[int, list[float]] = {}

    for trace_path in trace_files:
        rank = _extract_rank_from_filename(trace_path)

        try:
            with trace_path.open() as f:
                data = json.load(f)

            events = data.get("traceEvents", [])

            # Look for memory snapshot events
            for event in events:
                cat = event.get("cat", "")
                name = event.get("name", "")
                ph = event.get("ph", "")
                args = event.get("args", {})

                # Memory snapshots can be instant events or named events
                is_snapshot = (
                    (ph == "i" and ("memory" in name.lower() or "Memory" in name))
                    or "memory" in cat.lower()
                    or "memory" in name.lower()
                    or "Memory" in name
                )

                if is_snapshot:
                    # Try to get current memory usage from various field names
                    memory_fields = [
                        "Allocated Bytes",
                        "Reserved Bytes",
                        "allocated_bytes",
                        "reserved_bytes",
                        "current_allocated",
                        "current_reserved",
                        "Device Total Allocated",
                        "Device Total Reserved",
                        "total_allocated",
                        "total_reserved",
                        "size",
                        "bytes",
                    ]
                    for key in memory_fields:
                        if key in args:
                            size_bytes = args[key]
                            if isinstance(size_bytes, (int, float)) and size_bytes > 0:
                                memory_gb = size_bytes / (1024**3)
                                if rank not in memory_samples:
                                    memory_samples[rank] = []
                                memory_samples[rank].append(memory_gb)
                                break

            # Also check trace metadata for memory info
            metadata = data.get("metadata", {})
            if "memory" in metadata:
                mem_info = metadata["memory"]
                if isinstance(mem_info, dict):
                    for key in ["allocated_bytes", "reserved_bytes", "total_allocated"]:
                        if key in mem_info:
                            size_bytes = mem_info[key]
                            if isinstance(size_bytes, (int, float)) and size_bytes > 0:
                                memory_gb = size_bytes / (1024**3)
                                if rank not in memory_samples:
                                    memory_samples[rank] = []
                                memory_samples[rank].append(memory_gb)
                                break

        except Exception as e:
            print(f"  Error extracting memory from {trace_path.name}: {e}", file=sys.stderr)
            continue

    # Calculate peak and average
    for rank, samples in memory_samples.items():
        if samples:
            peak_memory_gb[rank] = max(samples)
            avg_memory_gb[rank] = sum(samples) / len(samples)

    return peak_memory_gb, avg_memory_gb


def _extract_rank_from_filename(path: Path) -> int:
    """Extract rank number from trace filename."""
    name = path.stem  # filename without extension

    # Look for "rank" followed by digits
    match = re.search(r"rank[_\s]*(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Look for just digits at the end
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))

    return 0  # Default to rank 0


def _calculate_mfu(trace_dir: Path, workload_card: dict[str, Any]) -> dict[str, Any]:
    """Calculate Model FLOP Utilization (MFU) and TFLOPs per GPU.

    MFU = (Actual FLOPS per GPU) / (Peak FLOPS per GPU)
    TFLOPs = Actual FLOPS per GPU / 1e12

    This requires:
    - Model architecture info (from workload card)
    - Actual compute time from traces
    - Peak FLOPS for the GPU model
    - Parallelism config (PP, TP) to account for model sharding

    Note: For PP and TP, each GPU only processes a fraction of the model,
    so we divide the full model FLOPs by (PP * TP) to get per-GPU FLOPs.
    DP doesn't reduce per-GPU FLOPs, it replicates computation.
    """
    # Extract model config
    workload = workload_card.get("workload", {})
    model_config = workload.get("model", {})
    data_config = workload.get("data", {})

    # Get model parameters (approximate)
    model_name = model_config.get("name", "")
    seq_len = data_config.get("seq_len", 0)
    batch_size = data_config.get("batch_size", 0)

    # Try alternative field names
    if seq_len == 0:
        seq_len = data_config.get("sequence_length", 4096)
    if batch_size == 0:
        batch_size = data_config.get("local_batch_size", data_config.get("micro_batch_size", 1))

    if seq_len == 0 or batch_size == 0:
        print(
            f"  Warning: seq_len={seq_len}, batch_size={batch_size} from workload card",
            file=sys.stderr,
        )
        return {
            "mfu_percent": 0.0,
            "tflops_per_gpu": 0.0,
            "note": f"Missing seq_len or batch_size (seq_len={seq_len}, batch_size={batch_size})",
        }

    # Try to get iteration time from traces
    iter_time_ms = _get_avg_iter_time(trace_dir)

    if iter_time_ms <= 0:
        print("  Warning: Could not determine iteration time from traces", file=sys.stderr)
        return {
            "mfu_percent": 0.0,
            "tflops_per_gpu": 0.0,
            "note": "Could not determine iteration time from traces",
        }

    print(
        f"  MFU calculation: seq_len={seq_len}, batch_size={batch_size}, "
        f"iter_time={iter_time_ms:.1f}ms",
        file=sys.stderr,
    )

    # Estimate model FLOPs per token
    estimated_flops_per_token = _estimate_flops_per_token(model_name, model_config)

    if estimated_flops_per_token <= 0:
        print(
            f"  Warning: Could not estimate FLOPs per token (model_name={model_name})",
            file=sys.stderr,
        )
        return {
            "mfu_percent": 0.0,
            "tflops_per_gpu": 0.0,
            "note": f"Could not estimate model FLOPs (model_name={model_name})",
        }

    print(f"  Estimated FLOPs per token: {estimated_flops_per_token:.2e}", file=sys.stderr)

    # Get parallelism config to account for model sharding
    parallel = workload_card.get("parallelism", {})
    pp = parallel.get("pp", 1)  # Pipeline Parallel degree
    tp = parallel.get("tp", 1)  # Tensor Parallel degree

    # Account for model sharding:
    # - PP: Each GPU processes 1/PP of the model layers
    # - TP: Each GPU processes 1/TP of the tensor operations
    # - Combined: Each GPU processes 1/(PP*TP) of the model FLOPs
    parallelism_factor = pp * tp
    if parallelism_factor == 0:
        parallelism_factor = 1  # Safety check

    print(f"  Parallelism: PP={pp}, TP={tp}, factor={parallelism_factor}", file=sys.stderr)

    # Calculate actual FLOPS per GPU
    tokens_per_iter = batch_size * seq_len
    flops_per_iter_full_model = estimated_flops_per_token * tokens_per_iter
    # Divide by parallelism factor to get per-GPU FLOPs
    flops_per_iter_per_gpu = flops_per_iter_full_model / parallelism_factor
    iter_time_s = iter_time_ms / 1000.0

    if iter_time_s <= 0:
        return {
            "mfu_percent": 0.0,
            "tflops_per_gpu": 0.0,
            "note": "Invalid iteration time",
        }

    flops_per_sec_per_gpu = flops_per_iter_per_gpu / iter_time_s
    print(f"  FLOPs per iter (full model): {flops_per_iter_full_model:.2e}", file=sys.stderr)
    print(f"  FLOPs per iter (per GPU): {flops_per_iter_per_gpu:.2e}", file=sys.stderr)
    print(f"  FLOPs per sec (per GPU): {flops_per_sec_per_gpu:.2e}", file=sys.stderr)

    # Get peak FLOPS for GPU (simplified - would need actual GPU model detection)
    peak_tflops = _get_peak_tflops(workload_card)

    # Calculate MFU (per GPU)
    if peak_tflops > 0:
        mfu = (flops_per_sec_per_gpu / 1e12) / peak_tflops * 100  # Convert to percentage
        # Cap MFU at 100%
        if mfu > 100:
            print(
                f"  Warning: MFU calculated as {mfu:.1f}%, capping at 100% "
                "(likely due to parallelism calculation or measurement error)",
                file=sys.stderr,
            )
            mfu = min(mfu, 100.0)
    else:
        mfu = 0.0

    # TFLOPs per GPU
    tflops_per_gpu = flops_per_sec_per_gpu / 1e12

    return {
        "mfu_percent": mfu,
        "tflops_per_gpu": tflops_per_gpu,
        "estimated_flops_per_token": estimated_flops_per_token,
        "peak_tflops_per_gpu": peak_tflops,
    }


def _get_avg_iter_time(trace_dir: Path) -> float:
    """Get average iteration time from traces."""
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
            with trace_path.open() as f:
                data = json.load(f)

            events = data.get("traceEvents", [])

            # Find ProfilerStep# events
            step_times = []
            for event in events:
                name = event.get("name", "")
                if name.startswith("ProfilerStep#") and event.get("ph") == "X":
                    dur = event.get("dur", 0)
                    if dur > 0:
                        step_times.append(dur / 1000.0)  # us -> ms

            if step_times:
                return sum(step_times) / len(step_times)

        except Exception:
            continue

    return 0.0


def _estimate_flops_per_token(model_name: str, model_config: dict[str, Any]) -> float:
    """Estimate FLOPs per token for a model.

    This is a simplified estimation. Real implementation would need:
    - Exact model architecture
    - Layer counts and sizes
    - Attention mechanism details
    """
    # Try to get parameter count
    num_params = model_config.get("num_params", 0)
    if num_params == 0:
        # Try to estimate from model name
        if "8b" in model_name.lower():
            num_params = 8e9
        elif "70b" in model_name.lower():
            num_params = 70e9
        elif "13b" in model_name.lower():
            num_params = 13e9
        else:
            return 0.0

    # Rough estimate for transformer training:
    # Forward pass: ~2 * num_params FLOPs per token
    # Backward pass: ~4 * num_params FLOPs per token (gradient computation)
    # Total: ~6 * num_params FLOPs per token
    return 6.0 * num_params


def _get_peak_tflops(workload_card: dict[str, Any]) -> float:
    """Get peak TFLOPs for the GPU model."""
    # Try to get from workload card
    hardware = workload_card.get("hardware", {})
    gpu_model = hardware.get("gpu_model", "").lower()

    # Common GPU peak TFLOPs (FP16/BF16)
    peak_tflops_map = {
        "a100": 312.0,  # A100 40GB/80GB
        "a100-40gb": 312.0,
        "a100-80gb": 312.0,
        "h100": 1000.0,  # H100
        "v100": 125.0,  # V100
        "a10": 125.0,  # A10
    }

    for key, tflops in peak_tflops_map.items():
        if key in gpu_model:
            return tflops

    # Default to A100 if unknown
    return 312.0


def _estimate_memory_from_model(workload_card: dict[str, Any]) -> float:
    """Estimate GPU memory usage from model parameters.

    Rough estimate:
    - Model weights: num_params * bytes_per_param (typically 2 bytes for bf16/fp16)
    - Activations: depends on batch_size, seq_len, hidden_size
    - Optimizer states: 2-3x model size for AdamW
    - Total: ~4-6x model size for training
    """
    workload = workload_card.get("workload", {})
    model_config = workload.get("model", {})
    data_config = workload.get("data", {})

    # Get model parameters
    num_params = model_config.get("num_params", 0)
    if num_params == 0:
        model_name = model_config.get("name", "")
        if "8b" in model_name.lower():
            num_params = 8e9
        elif "70b" in model_name.lower():
            num_params = 70e9
        elif "13b" in model_name.lower():
            num_params = 13e9
        else:
            return 0.0

    # Estimate memory:
    # - Model weights: num_params * 2 bytes (bf16/fp16)
    # - Optimizer states: num_params * 4 bytes (fp32 momentum + variance for AdamW)
    # - Gradients: num_params * 2 bytes (bf16/fp16)
    # - Activations: rough estimate based on batch_size and seq_len
    # Total: ~8-10 bytes per parameter for training

    bytes_per_param = 8.0  # Conservative estimate
    model_memory_gb = (num_params * bytes_per_param) / (1024**3)

    # Add activation memory estimate (rough)
    batch_size = data_config.get("batch_size", data_config.get("local_batch_size", 1))
    seq_len = data_config.get("seq_len", 4096)
    if batch_size > 0 and seq_len > 0:
        # Very rough: ~batch_size * seq_len * hidden_size * 2 bytes per activation
        # For 8B model, hidden_size ~ 4096
        hidden_size = model_config.get("hidden_size", 4096)
        activation_memory_gb = (batch_size * seq_len * hidden_size * 2) / (1024**3)
        model_memory_gb += activation_memory_gb

    return model_memory_gb

