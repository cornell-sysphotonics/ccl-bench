#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path

METRIC_TO_SCRIPT = {
    "TTFT": "TTFT-group_12/extract.py",
    "TPOT": "TPOT-group_12/extract.py",
    "request_throughput": "requestThroughput-group_12/extract.py",
    "output_throughput": "outputThroughput-group_12/extract.py",
    "token_to_expert_assignment": "tokenToExpertAssignment-group_12/extract.py",
    "communicationOverhead": "communicationOverhead-group_12/analyzeEPLB.py",
}

def main():
    parser = argparse.ArgumentParser(description="Extract metrics from trace directories")
    parser.add_argument("--trace", help="Path to trace directory (required for most metrics)")
    parser.add_argument("--metric", required=True, help="Name of metric to extract")
    parser.add_argument("--csv1", help="First CSV file (for communicationOverhead: EPLB OFF CSV)")
    parser.add_argument("--csv2", help="Second CSV file (for communicationOverhead: EPLB ON CSV)")
    
    args = parser.parse_args()
    
    if args.metric not in METRIC_TO_SCRIPT:
        print(f"Error: Unknown metric: {args.metric}", file=sys.stderr)
        print(f"Available metrics: {', '.join(METRIC_TO_SCRIPT.keys())}", file=sys.stderr)
        sys.exit(1)
    
    script_path = Path(__file__).parent / METRIC_TO_SCRIPT[args.metric]
    if not script_path.exists():
        print(f"Error: Metric script not found: {script_path}", file=sys.stderr)
        sys.exit(1)
    
    # Handle communicationOverhead metric which requires two CSV files
    if args.metric == "communicationOverhead":
        if not args.csv1 or not args.csv2:
            print("Error: communicationOverhead metric requires --csv1 and --csv2 arguments", file=sys.stderr)
            print("Usage: python main.py --metric communicationOverhead --csv1 <eplb_off.csv> --csv2 <eplb_on.csv>", file=sys.stderr)
            sys.exit(1)
        
        csv1_path = Path(args.csv1)
        csv2_path = Path(args.csv2)
        
        if not csv1_path.exists():
            print(f"Error: CSV file not found: {csv1_path}", file=sys.stderr)
            sys.exit(1)
        if not csv2_path.exists():
            print(f"Error: CSV file not found: {csv2_path}", file=sys.stderr)
            sys.exit(1)
        
        script_args = [str(csv1_path), str(csv2_path)]
    else:
        # Standard metrics that require a trace directory
        if not args.trace:
            print(f"Error: --trace argument required for metric: {args.metric}", file=sys.stderr)
            sys.exit(1)
        
        trace_dir = Path(args.trace)
        if not trace_dir.exists():
            print(f"Error: Trace directory not found: {trace_dir}", file=sys.stderr)
            sys.exit(1)
        
        script_args = [str(trace_dir)]
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + script_args,
            capture_output=True,
            text=True,
            check=False
        )
        
        sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running metric script: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

