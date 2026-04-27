#!/usr/bin/env python3
"""
Run bandwidth_utilization_allreduce_group_6.metric_cal on all trace directories
under /data/ccl-bench_trace_collection/ and print a summary.
"""

import sys
from pathlib import Path

# Make the tool importable
sys.path.insert(0, str(Path(__file__).parent / "bandwidth_utilization_allreduce_group_6"))
from bandwidth_utilization_allreduce_group_6 import metric_cal

TRACE_ROOT = Path("/data/ccl-bench_trace_collection")

results = []
dirs = sorted(d for d in TRACE_ROOT.iterdir() if d.is_dir())
print(f"Found {len(dirs)} directories\n")

for d in dirs:
    print(f"--- {d.name} ---")
    val = metric_cal(str(d))
    results.append((d.name, val))
    print(f"  => median allreduce BW: {val:.4f} GB/s\n" if val == val else f"  => no result\n")

print("\n=== Summary ===")
for name, val in results:
    status = f"{val:.4f} GB/s" if val == val else "N/A"
    print(f"  {name}: {status}")
