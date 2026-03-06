# configs

This directory holds the **source-of-truth runtime configs** for this workload.

Guidelines:
- Keep these files stable once you’ve collected traces.
- If you change anything (seq_len, DP/TP, nodes, etc.), create a NEW workload folder
  or commit the change with a clear tag (so traces remain attributable).

Files:
- `llama-3.1-8b-torchtitan-perlmutter.toml`: main configuration (production trace collection)
- `debug_model.toml`: smaller / debug configuration to validate the pipeline quickly
