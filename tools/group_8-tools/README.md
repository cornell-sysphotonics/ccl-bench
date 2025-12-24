# Straggler Delay Tool (Group 8)

This tool computes iteration-level timing imbalance (straggler effects) from
PyTorch Profiler JSON traces (Chrome trace format).

The tool was used in our Results section to:
1) compute per-iteration straggler delay and per-rank iteration end times; and
2) generate comparison plots between Experiment 1 and Experiment 2.

---

## Metric Definition

### Iteration End Time per Rank
For rank `i` at iteration `k`, we define the iteration end time as the latest
event end timestamp within the iteration window:

T_{i,k}^{end} = max_{e ∈ E_{i,k}} ( t_e^{start} + t_e^{dur} )

where each profiler event `e` provides `ts` (start timestamp) and `dur` (duration).

### Straggler Delay per Iteration
SD_k = max_i T_{i,k}^{end} − min_i T_{i,k}^{end}

### Straggler Ratio per Iteration
SR_k = SD_k / max_i T_{i,k}^{end}

---

## Input Format

- A directory containing per-rank PyTorch Profiler trace files:
  rank0_trace.json, rank1_trace.json, ...
- Each JSON file must contain a top-level `traceEvents` array with events
  containing `ts` and (optionally) `dur`.

---

## How to Run (Metric Computation)

From the repository root:

```bash
python3 tools/straggler_delay-group_8/straggler_delay-group_8.py \
  --dir <TRACE_DIR> \
  --pattern "rank*_trace.json" \
  --marker "ProfilerStep#" \
  --out_csv straggler_delay.csv \
  --out_summary summary.json
