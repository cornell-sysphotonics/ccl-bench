#!/usr/bin/env python3
"""
Parallelism configuration optimizer for distributed AI training.

Implements two policies:
  - baseline_policy:  hardcoded (tp=4, dp=2, pp=1)
  - learned_policy:   Claude agent that iteratively runs ASTRA-sim to find
                      the configuration minimizing wall time

Usage:
    python parallelism_agent.py [--config PATH] [--policy baseline|learned]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import anthropic
import yaml


# ── Paths ──────────────────────────────────────────────────────────────────────
AGENT_DIR       = Path(__file__).parent.parent.parent
API_KEY_FILE    = AGENT_DIR / "API_KEY"
DEFAULT_CONFIG  = Path(__file__).parent / "experiment.yaml"

# ── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a parallelism configuration optimizer for distributed AI training.

Your job is to find the (tp, dp, pp) parallelism policy that minimizes wall \
time for a given workload and hardware environment by running simulations with \
the `run_simulation` tool.

Guidelines:
- Valid configurations satisfy: tp * dp * pp ≤ total GPUs available (enforced \
  by the framework — configs exceeding the GPU count will be rejected before running). \
  You do not need to use all available GPUs.
- Use the `run_simulation` tool to evaluate candidate configurations.
- If a config returns an out-of-memory error, that configuration does not fit \
  on the hardware — try higher TP to shard weights/activations across more GPUs.
- Make informed choices about which configs to try based on simulation results — \
  you do not need to try every possible combination exhaustively.
- After sufficient exploration, output your recommendation as JSON on the last \
  line in the format:
      BEST_POLICY: {"tp": <int>, "dp": <int>, "pp": <int>, "wall_time": <int>}
"""


# ── Config loading ──────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _total_gpus(cfg: dict) -> int:
    env = cfg["environment"]
    return env["nodes"] * env["gpus_per_node"]


# ── Simulation tool ─────────────────────────────────────────────────────────────

def _update_network_yml(network_yml: Path, tp: int, dp: int, pp: int) -> None:
    """Update network.yml topology and npus_count to match the parallelism config.

    Maps parallelism dimensions to a two-tier network hierarchy:
      - Dimension 0 (Ring,  intra-node, fast):  tp
      - Dimension 1 (Switch, inter-node, slow):  dp * pp

    If total GPUs <= 4 a flat FullyConnected topology is used instead.
    """
    text = network_yml.read_text()

    n_gpus = tp * dp * pp
    if n_gpus <= 4:
        topology_str  = "[ FullyConnected ]"
        npus_str      = f"[ {n_gpus} ]"
        bandwidth_str = "[ 300 ]"
        latency_str   = "[ 500.0 ]"
    else:
        n_nodes       = n_gpus // 4
        topology_str  = "[ Ring, Switch ]"
        npus_str      = f"[ 4, {n_nodes} ]"
        bandwidth_str = "[ 300, 25 ]"
        latency_str   = "[ 500.0, 1000.0 ]"

    text = re.sub(r"topology:\s*\[.*?\]",   f"topology: {topology_str}",   text)
    text = re.sub(r"npus_count:\s*\[.*?\]", f"npus_count: {npus_str}",     text)
    text = re.sub(r"bandwidth:\s*\[.*?\]",  f"bandwidth: {bandwidth_str}", text)
    text = re.sub(r"latency:\s*\[.*?\]",    f"latency: {latency_str}",     text)
    network_yml.write_text(text)


def _parse_metrics(text: str) -> dict:
    """Parse ASTRA-sim output lines into per-rank metrics, returning max wall time."""
    if text.strip().startswith("out of memory"):
        return {
            "error": "out of memory",
            "message": (
                "Per-GPU memory exceeded the hardware limit. "
                "Try a configuration with higher TP (tensor parallelism) to shard "
                "weights/activations across more GPUs, or reduce DP/PP."
            ),
        }

    wall_times, gpu_times, comm_times = [], [], []
    for line in text.splitlines():
        if m := re.search(r"sys\[\d+\], Wall time: (\d+)", line):
            wall_times.append(int(m.group(1)))
        elif m := re.search(r"sys\[\d+\], GPU time: (\d+)", line):
            gpu_times.append(int(m.group(1)))
        elif m := re.search(r"sys\[\d+\], Comm time: (\d+)", line):
            comm_times.append(int(m.group(1)))

    if not wall_times:
        return {"error": "no metrics in output", "tail": text[-1500:]}

    avg = lambda xs: sum(xs) / len(xs)
    return {
        "wall_time":  max(wall_times),
        "gpu_time":   avg(gpu_times),
        "comm_time":  avg(comm_times),
        "num_ranks":  len(wall_times),
    }


def run_simulation(tp: int, dp: int, pp: int, cfg: dict) -> dict:
    """
    Run ASTRA-sim inside Docker with the given parallelism settings.

    Enforces constraints from cfg before invoking docker:
      - tp * dp * pp must equal environment.nodes * environment.gpus_per_node
      - environment.gpu_memory_gb is passed to run.sh via -m

    Returns a dict with wall_time (cycles), gpu_time, comm_time, num_ranks,
    or an 'error' key if the run failed or a constraint was violated.
    """
    # ── Hard constraint: GPU count ───────────────────────────────────────────
    total = _total_gpus(cfg)
    if tp * dp * pp > total:
        return {
            "error": "invalid config",
            "message": (
                f"tp*dp*pp={tp*dp*pp} exceeds total_gpus={total} "
                f"({cfg['environment']['nodes']} nodes × "
                f"{cfg['environment']['gpus_per_node']} GPUs/node). "
                f"Choose values whose product is ≤ {total}."
            ),
        }

    workload_name = cfg["simulation"]["workload_name"]
    env           = cfg["environment"]
    wl            = cfg["workload"]
    example_dir   = AGENT_DIR / "tools/astra-sim-hybrid-parallelism/examples" / workload_name
    network_yml   = example_dir / "network.yml"

    _update_network_yml(network_yml, tp, dp, pp)

    # Workload parameters forwarded as container environment variables.
    # Mapping: YAML key → env var name expected by run.sh
    workload_env = {
        "BATCH":          str(wl["batch_size"]),
        "SEQ":            str(wl["seq_len"]),
        "DMODEL":         str(wl["dmodel"]),
        "DFF":            str(wl["dff"]),
        "DVOCAL":         str(wl["dvocal"]),
        "HEAD":           str(wl["num_heads"]),
        "KVHEAD":         str(wl["num_kv_heads"]),
        "NUM_STACKS":     str(wl["num_stacks"]),
        "WEIGHT_SHARDED": str(wl["weight_sharded"]),
        "CHAKRA_VERSION": str(wl["chakra_schema_version"]),
    }

    env_flags = []
    for k, v in workload_env.items():
        env_flags += ["-e", f"{k}={v}"]

    cmd = [
        "docker", "run", "--rm",
        "--shm-size=8g",
        "-v", f"{AGENT_DIR}:/agent",
        "-w", f"/agent/tools/astra-sim-hybrid-parallelism/examples/{workload_name}",
        *env_flags,
        "astra-sim:latest",
        "bash", f"/agent/tools/astra-sim-hybrid-parallelism/examples/{workload_name}/run.sh",
        "-t", str(tp), "-d", str(dp), "-p", str(pp),
        "-m", str(env["gpu_memory_gb"]),   # GPU memory limit enforced inside run.sh
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"error": "simulation timed out after 600s"}
    except FileNotFoundError:
        return {"error": "docker not found — is Docker running?"}

    output_file = example_dir / "output.txt"
    raw = output_file.read_text() if output_file.exists() else (proc.stdout + proc.stderr)

    metrics = _parse_metrics(raw)
    if "error" in metrics and proc.returncode != 0:
        metrics["stderr"] = proc.stderr[-1000:]
    return metrics


# ── Policies ────────────────────────────────────────────────────────────────────

def baseline_policy(cfg: dict) -> dict:
    """Return the baseline configuration from the YAML."""
    return dict(cfg["baseline"])


def _build_user_message(cfg: dict) -> str:
    """Construct the LLM user message from the structured YAML config."""
    total = _total_gpus(cfg)
    cfg_yaml = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    return (
        f"Here is the full experiment configuration:\n\n"
        f"```yaml\n{cfg_yaml}```\n\n"
        f"Total available GPUs: {total} "
        f"({cfg['environment']['nodes']} nodes × {cfg['environment']['gpus_per_node']} GPUs/node).\n\n"
        f"Find the (tp, dp, pp) configuration that minimizes {cfg['optimization_goal']}. "
        f"Configurations with tp * dp * pp > {total} will be rejected."
    )


def learned_policy(cfg: dict, max_iterations: int = 10) -> dict:
    """
    Claude-driven policy.  Explores the parallelism search space using
    run_simulation as a tool and returns the (tp, dp, pp) configuration
    with the lowest wall time.
    """
    api_key = API_KEY_FILE.read_text().strip()
    client  = anthropic.Anthropic(api_key=api_key)

    total = _total_gpus(cfg)

    tools = [
        {
            "name": "run_simulation",
            "description": (
                "Run ASTRA-sim in a Docker container with the specified parallelism "
                f"configuration (must satisfy tp * dp * pp ≤ {total}) and return "
                "performance metrics. network.yml is automatically updated to match "
                "the GPU count. Returns an error if the config is invalid or OOM."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tp": {"type": "integer", "description": "Tensor parallelism degree (≥1)"},
                    "dp": {"type": "integer", "description": "Data parallelism degree (≥1)"},
                    "pp": {"type": "integer", "description": "Pipeline parallelism degree (≥1)"},
                },
                "required": ["tp", "dp", "pp"],
            },
        }
    ]

    tool_choice = {"type": "auto", "disable_parallel_tool_use": True}

    messages = [{"role": "user", "content": _build_user_message(cfg)}]
    best = None
    iteration = 0

    print(f"[learned_policy] total_gpus={total}, "
          f"gpu_memory={cfg['environment']['gpu_memory_gb']} GB, "
          f"max_iterations={max_iterations}")

    while iteration < max_iterations:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude] {block.text}")

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    m = re.search(r"BEST_POLICY:\s*(\{[^\}]+\})", block.text)
                    if m:
                        best = json.loads(m.group(1))
            break

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            break

        iteration += 1
        tp, dp, pp = tool_use.input["tp"], tool_use.input["dp"], tool_use.input["pp"]
        print(f"\n[sim {iteration}/{max_iterations}] tp={tp}, dp={dp}, pp={pp}  ({tp*dp*pp} GPUs)")
        result = run_simulation(tp, dp, pp, cfg)
        print(f"  → {result}")

        if "wall_time" in result:
            if best is None or result["wall_time"] < best.get("wall_time", float("inf")):
                best = {"tp": tp, "dp": dp, "pp": pp, "wall_time": result["wall_time"]}

        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": json.dumps(result)}],
        })

    # Ask Claude for its final recommendation if we hit the iteration cap
    if best and "wall_time" in best and not any(
        block.type == "text" and "BEST_POLICY" in (block.text or "")
        for turn in messages if turn["role"] == "assistant"
        for block in (turn["content"] if isinstance(turn["content"], list) else [])
    ):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "none"},
            messages=messages + [{"role": "user", "content": "Iteration budget exhausted. State your BEST_POLICY recommendation."}],
        )
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude] {block.text}")
                m = re.search(r"BEST_POLICY:\s*(\{[^\}]+\})", block.text)
                if m:
                    best = json.loads(m.group(1))

    baseline = baseline_policy(cfg)
    return best or baseline


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallelism policy optimizer using ASTRA-sim")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Path to experiment YAML config (default: experiment.yaml)"
    )
    parser.add_argument(
        "--policy", choices=["baseline", "learned"], default="learned",
        help="Which policy to run (default: learned)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Max simulation calls for the learned policy (default: 10)"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    env = cfg["environment"]
    wl  = cfg["workload"]
    total = _total_gpus(cfg)

    print("=" * 60)
    print(f"Policy:      {args.policy}")
    print(f"Workload:    {wl['model']}  batch={wl['batch_size']}  seq={wl['seq_len']}")
    print(f"Environment: {total} GPUs ({env['nodes']}n × {env['gpus_per_node']}g), "
          f"{env['gpu_model']} {env['gpu_memory_gb']} GB")
    print(f"Goal:        {cfg['optimization_goal']}")
    print("=" * 60)

    if args.policy == "baseline":
        policy = baseline_policy(cfg)
        print(f"\nBaseline policy: {policy}")
    else:
        policy = learned_policy(cfg, max_iterations=args.max_iterations)
        baseline = baseline_policy(cfg)
        print("\n" + "=" * 60)
        print(f"Baseline:       tp={baseline['tp']}, dp={baseline['dp']}, pp={baseline['pp']}")
        print(f"Learned policy: tp={policy['tp']}, dp={policy['dp']}, pp={policy['pp']}", end="")
        if "wall_time" in policy:
            print(f"  (wall_time={policy['wall_time']:,} cycles)")
        else:
            print()
        print("=" * 60)

    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
