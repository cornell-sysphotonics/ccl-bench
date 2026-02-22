# Experiment Configuration

This experiment benchmarks the Mixtral-8x7B-v0.1 model with the following configuration:

## Controlled Variables

- **Model Type**: Baseline
- **MAX_SEQS**: 8
- **BATCH_TOKENS**: 16384

## How to Run

1. **Start the server** (in one terminal on the compute node):
   ```bash
   ./server.sh
   ```

2. **Run the client** (in another terminal on the same node):
   ```bash
   ./client.sh
   ```

The server will start the vLLM inference server with the specified configuration, and the client will send benchmark requests and collect performance metrics.

Results will be saved in the `results_json/` directory.

Trace output and server logs will be saved to the `logs/` directory.
