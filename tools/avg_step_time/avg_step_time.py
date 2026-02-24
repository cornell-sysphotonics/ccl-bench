import json
from typing import List
import os
import statistics


def metric_cal(directory: str) -> float:
    """
    Calculate average step time (inference: decode of the batch, training: forward + backward)

    Args:
        directory (str): Path to the directory containing PyTorch ET trace JSON files.

    Returns:
        float: Average step time in seconds.
    """

    trace_file = None
    for file in os.listdir(directory):
        if file.endswith(".json"):
            trace_file = os.path.join(directory, file)
            break

    if trace_file is None:
        raise FileNotFoundError(f"No JSON file found in directory: {directory}")


    with open(trace_file, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # ── 1. Identify step events ────────────────────────────────────────────────
    # The main engine step is traced as '$core.py:331 step' (phase 'X' = complete event)
    STEP_EVENT_NAME = "$core.py:331 step"
    step_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("name", "") == STEP_EVENT_NAME
    ]

    num_steps = len(step_events)
    # print(f"{'='*55}")
    # print(f"  Trace file : {trace_file}")
    # print(f"  Step event : {STEP_EVENT_NAME}")
    # print(f"  Total steps: {num_steps}")
    # print(f"{'='*55}")

    if num_steps == 0:
        print("No step events found.")
        return

    # ── 2. Per-step latencies (duration in µs → ms) ────────────────────────────
    latencies_us = [e["dur"] for e in step_events]
    latencies_ms = [d / 1_000 for d in latencies_us]

    # print(f"\n{'Step':>6}  {'Start (µs)':>14}  {'Duration (µs)':>15}  {'Duration (ms)':>14}")
    # print("-" * 57)
    # for i, (e, lat_ms) in enumerate(zip(step_events, latencies_ms), start=1):
    #     # Classify step type heuristically: first step is typically prefill
    #     label = "prefill" if i == 1 else "decode"
    #     print(f"{i:>4} ({label:<7})  {e['ts']:>14.2f}  {e['dur']:>15.2f}  {lat_ms:>14.2f}")

    # ── 3. Summary statistics ──────────────────────────────────────────────────
    avg_ms   = statistics.mean(latencies_ms[1:-1])
    total_ms = sum(latencies_ms)

    # print(f"\n{'─'*55}")
    # print(f"  Average step latency : {avg_ms:.3f} ms")
    # print(f"  Total latency        : {total_ms:.3f} ms")

    # if num_steps > 1:
    #     prefill_ms = latencies_ms[0]
    #     decode_ms  = latencies_ms[1:]
    #     avg_decode = statistics.mean(decode_ms)
    #     print(f"\n  Prefill latency      : {prefill_ms:.3f} ms")
    #     print(f"  Avg decode latency   : {avg_decode:.3f} ms  ({len(decode_ms)} step(s))")

    # print(f"{'─'*55}")

    # ── 4. Additional context: TPU kernel breakdown ────────────────────────────
    # The main model kernel during prefill
    # model_kernels = [
    #     e for e in events
    #     if e.get("ph") == "X"
    #     and e.get("name", "").startswith("jit_run_model")
    # ]
    # if model_kernels:
    #     print(f"\n  TPU jit_run_model kernels found: {len(model_kernels)}")
    #     for k in model_kernels:
    #         print(f"    run_id={k['args'].get('run_id','?')}  "
    #               f"device_dur={int(k['args'].get('device_duration_ps',0))/1e9:.3f} ms  "
    #               f"host_dur={k['dur']/1000:.3f} ms")

    return avg_ms / 1000  # Convert ms to seconds