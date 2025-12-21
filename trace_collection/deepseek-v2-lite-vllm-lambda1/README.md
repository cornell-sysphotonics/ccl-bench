# DeepSeek-V2-Lite (vLLM Lambda1)

## Implementation Details
This workload profiles the NVLink interconnect utilization for the DeepSeek-V2-Lite model using the vLLM serving engine on the Lambda1 platform.

- **Framework:** vLLM
- **Model:** DeepSeek-V2-Lite
- **Parallelism:** EP=2 (Expert Parallelism)
- **Platform:** Lambda1

## Execution Method
This experiment was conducted using the `serve_deepseek_v2.sh` script from the vllm-project examples.

## Artifacts Generated
- `deepseek-v2-lite-vllm-lambda1.yaml`: Workload card with experiment metadata.
- Traces collected include PyTorch ET and hardware metrics.

