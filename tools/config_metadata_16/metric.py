"""Config Metadata metric calculation.

Extracts parallelism configuration and hardware information from workload cards
and trace metadata. This provides a summary table for comparing runs.

NVTX Dependency: None
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
    """Extract configuration and metadata from workload card and traces.

    Args:
        directory: Path to the trace directory containing workload_card.yaml
                   and trace files.
        profile_mode: Profile mode - 'torch', 'nsys', or 'auto' (detect).
        workload_card_path: Optional explicit path to workload card YAML or TOML file.

    Returns:
        Dictionary with configuration metadata:
            - parallelism: Dict with world_size, tp, dp, pp, cp, etc.
            - data: Dict with seq_len, batch_size, global_batch_size, etc.
            - hardware: Dict with gpu_model, num_gpus, interconnect, etc.
            - model: Dict with model name, num_params, etc.
    """
    trace_dir = Path(directory)

    # Load workload card - use explicit path if provided, otherwise search
    if workload_card_path:
        workload_card = _load_workload_card_from_path(Path(workload_card_path))
    else:
        workload_card = _load_workload_card(trace_dir)

    result: dict[str, Any] = {
        "parallelism": {},
        "data": {},
        "hardware": {},
        "model": {},
    }

    if workload_card:
        # Extract parallelism config
        parallel = workload_card.get("parallelism", {})
        result["parallelism"] = {
            "world_size": parallel.get("world_size", 0),
            "tp": parallel.get("tp", 1),
            "dp": parallel.get("dp", 1),
            "pp": parallel.get("pp", 1),
            "cp": parallel.get("cp", 1),
            "ep": parallel.get("ep", 1),
            "dp_shard": parallel.get("dp_shard", 1),
            "dp_repl": parallel.get("dp_repl", 1),
        }

        # Debug output
        if result["parallelism"].get("tp", 1) > 1 or result["parallelism"].get("pp", 1) > 1:
            print(
                f"  Config metadata: TP={result['parallelism'].get('tp')}, "
                f"PP={result['parallelism'].get('pp')}, "
                f"CP={result['parallelism'].get('cp')}",
                file=sys.stderr,
            )

        # Extract data config
        workload = workload_card.get("workload", {})
        data_config = workload.get("data", {})
        result["data"] = {
            "seq_len": data_config.get("seq_len", 0),
            "batch_size": data_config.get("batch_size", 0),
            "global_batch_size": data_config.get("global_batch_size", 0),
            "local_batch_size": data_config.get("local_batch_size", 0),
            "micro_batch_size": data_config.get("micro_batch_size", 0),
        }

        # Extract hardware config
        hardware = workload_card.get("hardware", {})
        result["hardware"] = {
            "gpu_model": hardware.get("gpu_model", "unknown"),
            "num_gpus": hardware.get("num_gpus", 0),
            "interconnect": hardware.get("interconnect", "unknown"),
            "gpu_memory_gb": hardware.get("gpu_memory_gb", 0),
        }

        # Extract model config
        model_config = workload.get("model", {})
        result["model"] = {
            "name": model_config.get("name", "unknown"),
            "num_params": model_config.get("num_params", 0),
            "hidden_size": model_config.get("hidden_size", 0),
            "num_layers": model_config.get("num_layers", 0),
            "num_heads": model_config.get("num_heads", 0),
        }

    # Try to infer from trace files if workload card is missing
    if not workload_card:
        print(
            f"  Warning: No workload card (YAML or TOML) found in {trace_dir} "
            "or parent directories",
            file=sys.stderr,
        )
        inferred = _infer_from_traces(trace_dir)
        if inferred:
            # Merge inferred data
            for key in ["parallelism", "data", "hardware", "model"]:
                if key in inferred and not result[key]:
                    result[key] = inferred[key]
                elif key in inferred:
                    result[key].update(inferred[key])
    elif not result["parallelism"].get("world_size"):
        print(
            "  Warning: Workload card found but world_size is 0. "
            "Parallelism config may be missing.",
            file=sys.stderr,
        )

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
        elif card_path.suffix.lower() == ".toml" and HAS_TOML and tomllib is not None:
            with card_path.open("rb") as f:
                data = tomllib.load(f)
                if data:
                    converted = _convert_toml_to_workload_card(data)
                    if converted:
                        print(f"Loaded TOML config: {card_path}", file=sys.stderr)
                        return converted
        else:
            print(f"Warning: Unsupported file type: {card_path.suffix}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading {card_path}: {e}", file=sys.stderr)

    return None


def _load_workload_card(trace_dir: Path) -> dict[str, Any] | None:
    """Load workload card YAML or TOML from directory and parent directories."""
    # Try YAML files first
    yaml_names = [
        "workload_card.yaml",
        "workload_card_tp.yaml",
        "workload_card_pp.yaml",
        "workload_card_dp_tp.yaml",
        "workload_card_dp_pp.yaml",
        "workload_card_3d.yaml",
    ]

    # Check current directory, parent, and grandparent
    dirs_to_check = [trace_dir, trace_dir.parent, trace_dir.parent.parent]

    for check_dir in dirs_to_check:
        # Try YAML files
        for name in yaml_names:
            card_path = check_dir / name
            if card_path.exists():
                try:
                    with card_path.open() as f:
                        data = yaml.safe_load(f)
                        if data:
                            print(f"Loaded workload card: {card_path}", file=sys.stderr)
                            return data
                except Exception as e:
                    print(f"Error loading {card_path}: {e}", file=sys.stderr)

        # Try TOML files (look for train_configs directory or any .toml file)
        if HAS_TOML and tomllib is not None:
            # Check for train_configs subdirectory
            train_configs_dir = check_dir / "train_configs"
            if train_configs_dir.exists():
                for toml_file in train_configs_dir.glob("*.toml"):
                    try:
                        with toml_file.open("rb") as f:
                            data = tomllib.load(f)
                            if data:
                                # Convert TOML structure to workload card format
                                converted = _convert_toml_to_workload_card(data)
                                if converted:
                                    print(f"Loaded TOML config: {toml_file}", file=sys.stderr)
                                    return converted
                    except Exception as e:
                        print(f"Error loading {toml_file}: {e}", file=sys.stderr)

            # Also check for .toml files directly in the directory
            for toml_file in check_dir.glob("*.toml"):
                try:
                    with toml_file.open("rb") as f:
                        data = tomllib.load(f)
                        if data:
                            converted = _convert_toml_to_workload_card(data)
                            if converted:
                                print(f"Loaded TOML config: {toml_file}", file=sys.stderr)
                                return converted
                except Exception as e:
                    print(f"Error loading {toml_file}: {e}", file=sys.stderr)

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


def _infer_from_traces(trace_dir: Path) -> dict[str, Any] | None:
    """Try to infer configuration from trace files."""
    inferred: dict[str, Any] = {
        "parallelism": {},
        "data": {},
        "hardware": {},
        "model": {},
    }

    # Count trace files to infer world_size
    trace_patterns = [
        "kineto_trace*.json",
        "rank*_trace.json",
        "*trace*.json",
    ]

    trace_files: set[Path] = set()
    for pattern in trace_patterns:
        trace_files.update(trace_dir.glob(pattern))

    profile_dir = trace_dir / "profile_trace"
    if profile_dir.exists():
        for pattern in trace_patterns:
            trace_files.update(profile_dir.glob(pattern))

    # Count unique ranks
    ranks: set[int] = set()
    for trace_path in trace_files:
        if trace_path.is_file() and trace_path.suffix == ".json":
            rank = _extract_rank_from_filename(trace_path)
            ranks.add(rank)

    if ranks:
        inferred["parallelism"]["world_size"] = len(ranks)

    # Try to extract hardware info from trace metadata
    for trace_path in trace_files:
        if not trace_path.is_file() or trace_path.suffix != ".json":
            continue

        try:
            with trace_path.open() as f:
                data = json.load(f)

            # Check metadata
            metadata = data.get("metadata", {})
            if "gpu_model" in metadata:
                inferred["hardware"]["gpu_model"] = metadata["gpu_model"]
            if "device_name" in metadata:
                inferred["hardware"]["gpu_model"] = metadata["device_name"]

            # Check event args for hardware info
            events = data.get("traceEvents", [])
            for event in events:
                args = event.get("args", {})
                if "gpu_model" in args and not inferred["hardware"].get("gpu_model"):
                    inferred["hardware"]["gpu_model"] = args["gpu_model"]
                if "device_name" in args and not inferred["hardware"].get("gpu_model"):
                    inferred["hardware"]["gpu_model"] = args["device_name"]

        except Exception:
            continue

    return inferred if any(inferred.values()) else None


def _extract_rank_from_filename(path: Path) -> int:
    """Extract rank number from trace filename."""
    name = path.stem

    match = re.search(r"rank[_\s]*(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))

    return 0

