#!/usr/bin/env python3
"""
Calculate TPOT (Time-per-Output-Token) metric.

TPOT is a runtime measurement metric that cannot be extracted from trace files.
Use the measurement scripts in the scripts/ directory instead.
"""


def tpot(directory: str) -> str:
    """
    TPOT metric calculation - requires runtime measurement.

    Args:
        directory (str): Path to directory (not used, kept for interface compatibility).

    Returns:
        str: Error message directing user to use measurement scripts.
    """
    import json
    
    error_msg = (
        "TPOT (Time-per-Output-Token) is a runtime measurement metric that cannot be "
        "extracted from trace files.\n\n"
        "To measure TPOT, please use the measurement scripts:\n"
        "  ./scripts/get_tpot.sh <model> <tp_size> [options]\n\n"
        "Or run the measurement tool directly:\n"
        "  python3 scripts/measure_tpot.py --model <model> --tp-size <tp_size>\n\n"
        "See scripts/README.md for detailed usage instructions."
    )
    
    return json.dumps({
        "error": error_msg,
        "metric": "tpot",
        "note": "TPOT requires runtime measurement, not trace extraction"
    }, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate TPOT metric")
    parser.add_argument("directory", type=str, help="Path to directory")
    args = parser.parse_args()
    
    result = tpot(args.directory)
    print(result)
