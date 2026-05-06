# CCL-Bench Agent Tools

## ASTRA-sim

The astra-sim-hybrid-parallelism package lives at `plug-ins/astra-sim-hybrid-parallelism` in the repo root. It is a fork of [ASTRA-sim](https://astra-sim.github.io/), a distributed AI system simulator that models the end-to-end software and hardware stack of modern AI systems.

### Building the Docker image

From the `plug-ins/astra-sim-hybrid-parallelism` directory:

```bash
cd plug-ins/astra-sim-hybrid-parallelism
docker build -t astra-sim:latest -f Dockerfile .
```

### Running the container with the plug-ins folder mounted

All commands use the absolute path to `plug-ins/` to avoid ambiguity regardless of working directory. Set `PLUGINS_DIR` once:

```bash
PLUGINS_DIR=/home/dd687/ccl-bench/plug-ins
```

Interactive shell:

```bash
PLUGINS_DIR=/home/dd687/ccl-bench/plug-ins
docker run -it --name astra-sim \
  --shm-size=8g \
  -v "$PLUGINS_DIR:/plug-ins" \
  astra-sim:latest bash
```

The `plug-ins/` directory will be available at `/plug-ins` inside the container. Files edited on the host are immediately visible inside the container and vice versa.

### Running examples directly from outside the container

Because `plug-ins/` is mounted at `/plug-ins`, scripts can be executed directly via `docker run` without entering an interactive shell. The ASTRA-sim binaries live in the image at `/app/astra-sim` while the scripts and configs are read from the host mount.

**Run the Llama example (one-shot):**

```bash
PLUGINS_DIR=/home/dd687/ccl-bench/plug-ins

# Flags: -t <tensor_parallel> -d <data_parallel> -p <pipeline_parallel>
docker run --rm \
  --shm-size=8g \
  -v "$PLUGINS_DIR:/plug-ins" \
  -w /plug-ins/astra-sim-hybrid-parallelism/examples/llama \
  astra-sim:latest \
  bash /plug-ins/astra-sim-hybrid-parallelism/examples/llama/run.sh \
  -t 4 -d 2 -p 1
```

`output.txt` will appear at `plug-ins/astra-sim-hybrid-parallelism/examples/llama/output.txt` on the host.

**Workflow for agents:**

1. Edit `run.sh` or any config (e.g. `system.json`, `network.yml`) on the host inside `plug-ins/astra-sim-hybrid-parallelism/examples/llama/`.
2. Re-run the `docker run` command above — no rebuild needed.
3. Read `output.txt` from the host to inspect results.

### First-time setup (submodules)

If the submodules under `astra-sim-hybrid-parallelism` are not yet initialized, run this from the repo root before building:

```bash
git submodule update --init --recursive
```
