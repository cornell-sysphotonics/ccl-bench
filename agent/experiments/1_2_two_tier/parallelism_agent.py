#!/usr/bin/env python3
"""
Parallelism configuration optimizer for distributed AI training.

Implements two policies:
  - baseline_policy:  hardcoded (tp=4, dp=2, pp=1)
  - learned_policy:   Claude agent that iteratively runs ASTRA-sim to find
                      the configuration minimizing wall time

Usage:
    python parallelism_agent_1.py [--prompt PATH] [--policy baseline|learned]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import anthropic

# TODO edit the network.yml config process


# ── Paths ──────────────────────────────────────────────────────────────────────
AGENT_DIR    = Path(__file__).parent.parent
EXAMPLE_DIR  = AGENT_DIR / "tools/astra-sim-hybrid-parallelism/examples/llama_2"
NETWORK_YML  = EXAMPLE_DIR / "network.yml"
API_KEY_FILE = AGENT_DIR / "API_KEY"
DEFAULT_PROMPT = Path(__file__).parent / "exp_agent_prompt.txt"

# ── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a parallelism configuration optimizer for distributed AI training.

Your job is to find the (tp, dp, pp) parallelism policy that minimizes wall \
time for a given workload and hardware environment by running simulations with \
the `run_simulation` tool.

Guidelines:
- Valid configurations satisfy: tp * dp * pp = total GPUs available. Start by \
  reasoning about how many GPUs the environment has.
- Use the `run_simulation` tool to evaluate candidate configurations.
- Make informed choices about which configs to try based on simulation results — \
  you do not need to try every possible combination exhaustively.
- After sufficient exploration, output your recommendation as JSON on the last \
  line in the format:
      BEST_POLICY: {"tp": <int>, "dp": <int>, "pp": <int>, "wall_time": <int>}
"""

# ── Simulation tool ─────────────────────────────────────────────────────────────

def _update_network_yml(n_gpus: int) -> None:
    """Set npus_count in network.yml to n_gpus."""
    text = NETWORK_YML.read_text()
    updated = re.sub(r"npus_count:\s*\[\s*\d+\s*\]", f"npus_count: [ {n_gpus} ]", text)
    NETWORK_YML.write_text(updated)


def _parse_metrics(text: str) -> dict:
    """Parse ASTRA-sim output lines into per-rank metrics, returning max wall time."""
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
        "wall_time":  max(wall_times),           # critical-path rank
        "gpu_time":   avg(gpu_times),
        "comm_time":  avg(comm_times),
        "num_ranks":  len(wall_times),
    }


def run_simulation(tp: int, dp: int, pp: int) -> dict:
    """
    Run ASTRA-sim inside Docker with the given parallelism settings.
    Updates network.yml so npus_count matches tp*dp*pp, then launches the
    container and parses the resulting output.txt.

    Returns a dict with wall_time (cycles), gpu_time, comm_time, num_ranks,
    or an 'error' key if the run failed.
    """
    n_gpus = tp * dp * pp
    _update_network_yml(n_gpus)

    cmd = [
        "docker", "run", "--rm",
        "--shm-size=8g",
        "-v", f"{AGENT_DIR}:/agent",
        "-w", "/agent/tools/astra-sim-hybrid-parallelism/examples/llama",
        "astra-sim:latest",
        "bash", "/agent/tools/astra-sim-hybrid-parallelism/examples/llama/run.sh",
        "-t", str(tp), "-d", str(dp), "-p", str(pp),
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return {"error": "simulation timed out after 600s"}
    except FileNotFoundError:
        return {"error": "docker not found — is Docker running?"}

    output_file = EXAMPLE_DIR / "output.txt"
    raw = output_file.read_text() if output_file.exists() else (proc.stdout + proc.stderr)

    metrics = _parse_metrics(raw)
    if "error" in metrics and proc.returncode != 0:
        metrics["stderr"] = proc.stderr[-1000:]
    return metrics


# ── Policies ────────────────────────────────────────────────────────────────────

def baseline_policy(workload: str, environment: str) -> dict:
    """Hardcoded baseline: tp=4, dp=2, pp=1."""
    return {"tp": 4, "dp": 2, "pp": 1}


def learned_policy(prompt_text: str, max_iterations: int = 10) -> dict:
    """
    Claude-driven policy.  Reads the experiment prompt, explores the
    parallelism search space using run_simulation as a tool, and returns
    the (tp, dp, pp) configuration with the lowest wall time.

    Each iteration Claude proposes exactly one simulation; the result is
    fed back immediately before the next proposal.  Stops after
    max_iterations simulations or when Claude emits BEST_POLICY.
    """
    api_key = API_KEY_FILE.read_text().strip()
    client  = anthropic.Anthropic(api_key=api_key)

    tools = [
        {
            "name": "run_simulation",
            "description": (
                "Run ASTRA-sim in a Docker container with the specified parallelism "
                "configuration and return performance metrics. "
                "network.yml is automatically updated to match tp*dp*pp GPUs."
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

    # auto + disable_parallel_tool_use: Claude calls at most one tool per turn,
    # but can also choose to stop — compatible with thinking (unlike "any").
    tool_choice = {"type": "auto", "disable_parallel_tool_use": True}

    messages = [{"role": "user", "content": prompt_text}]
    best = None
    iteration = 0

    print(f"\n[learned_policy] Starting exploration (max {max_iterations} iterations)...")

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

        # Append assistant turn to conversation history
        messages.append({"role": "assistant", "content": response.content})

        # Print any text blocks
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude] {block.text}")

        # Claude chose to stop — extract recommendation
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    m = re.search(r"BEST_POLICY:\s*(\{[^\}]+\})", block.text)
                    if m:
                        best = json.loads(m.group(1))
            break

        # Execute the single tool call and feed result back immediately
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            break

        iteration += 1
        tp, dp, pp = tool_use.input["tp"], tool_use.input["dp"], tool_use.input["pp"]
        print(f"\n[sim {iteration}/{max_iterations}] tp={tp}, dp={dp}, pp={pp}  ({tp*dp*pp} GPUs)")
        result = run_simulation(tp, dp, pp)
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

    return best or {"tp": 4, "dp": 2, "pp": 1}  # fallback to baseline


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallelism policy optimizer using ASTRA-sim")
    parser.add_argument(
        "--prompt", default=str(DEFAULT_PROMPT),
        help="Path to experiment prompt file (default: exp_agent_prompt.txt)"
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

    prompt_text = Path(args.prompt).read_text()

    # Parse workload/environment from prompt (simple extraction)
    env_match      = re.search(r"Environment:\s*(.+?)(?=\n\n|\Z)", prompt_text, re.DOTALL)
    workload_match = re.search(r"Workload:\s*(.+?)(?=\n\n|\Z)", prompt_text, re.DOTALL)
    environment = env_match.group(1).strip()    if env_match    else ""
    workload    = workload_match.group(1).strip() if workload_match else ""

    print("=" * 60)
    print(f"Policy:      {args.policy}")
    print(f"Workload:    {workload[:80]}")
    print(f"Environment: {environment[:80]}")
    print("=" * 60)

    if args.policy == "baseline":
        policy = baseline_policy(workload, environment)
        print(f"\nBaseline policy: {policy}")
    else:
        policy = learned_policy(prompt_text, max_iterations=args.max_iterations)
        baseline = baseline_policy(workload, environment)
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
