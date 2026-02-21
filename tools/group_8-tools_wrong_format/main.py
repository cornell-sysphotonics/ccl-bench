# in tools/main.py
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("straggler_delay-group_8")
    p.add_argument("--dir", required=True)
    p.add_argument("--pattern", default="rank*_trace.json")
    p.add_argument("--marker", default="ProfilerStep#")
    p.add_argument("--out_csv", default="straggler_delay.csv")
    p.add_argument("--out_summary", default="summary.json")

    args = ap.parse_args()

    if args.cmd == "straggler_delay-group_8":
        script = Path(__file__).parent / "straggler_delay-group_8" / "straggler_delay-group_8.py"
        cmd = [
            sys.executable, str(script),
            "--dir", args.dir,
            "--pattern", args.pattern,
            "--marker", args.marker,
            "--out_csv", args.out_csv,
            "--out_summary", args.out_summary,
        ]
        raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
