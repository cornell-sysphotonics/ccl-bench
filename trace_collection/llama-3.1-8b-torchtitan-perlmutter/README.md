# Llama-3.1-8B (TorchTitan Perlmutter)

## Implementation Details
This workload profiles the NVLink interconnect utilization for the Llama-3.1-8B model using the TorchTitan framework on the Perlmutter HPC cluster.

- **Framework:** TorchTitan
- **Model:** Llama-3.1-8B
- **Parallelism:** TP=4, PP=2, DP=2
- **Platform:** Perlmutter

## Execution Method
The experiment was conducted by modifying the `train.py` script in the TorchTitan repository to include `nsys` and custom tracing hooks.

## Artifacts Generated
- `llama-3.1-8b-torchtitan-perlmutter.yaml`: Workload card with experiment metadata.
- Traces collected include `nsys`, `torch_et`, and `kineto_trace`.

