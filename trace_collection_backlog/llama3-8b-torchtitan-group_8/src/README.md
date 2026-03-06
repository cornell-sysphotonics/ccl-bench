# src

Python code used by the launchers.

- `profiling.py`: enables/disables profiling, handles export, trace folder naming, and frequency.
- `logging.py`: small helpers to keep logs consistent across ranks.
- `utils.py`: common utilities (arg parsing, path helpers, formatting, etc.)

These scripts are intentionally lightweight so the trace collection pipeline is easy to audit.
