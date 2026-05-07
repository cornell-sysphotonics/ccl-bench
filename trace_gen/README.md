# Trace generation

Trace definition: Arjun, Abhishek

Trace collection method: Eric

## Collection window

Collect traces from steady-state execution, not from startup or first-use compilation.

For training workloads, run at least five warm-up iterations first so CUDA kernels
can compile and memory allocation can settle [55]. Then collect at least five
training iterations, or steps, per workload in steady state. Record the collected
iteration count in the workload card so downstream analyses can quantify the
variance behind a published number.

For inference workloads, first send a small number of warm-up batches so the
serving engine completes CUDA graph capture and KV-cache initialization [70].
Then collect traces that cover both the prefill pass, meaning prompt processing,
and at least 128 steady-state decode steps, meaning autoregressive token
generation.

Contributors are not required to perform full workload training or fine-tuning.
CCL-Bench does not collect accuracy metrics.

## GPU
### Current trace format
1. torch_et_\<rank>.json
2. kineto_trace_\<rank>.json
3. nsys_\<rank>.nsys-rep
4. metric specific trace

Trace collection method (from [Chakra Execution Trace Collection](https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces) guide)

This section focuses on simultaneous collection methods for PyTorch execution traces and Kineto traces.
### Collecting PyTorch Execution Traces
You can collect PyTorch execution traces from a PyTorch model's execution. This is achieved by using the [ExecutionTraceObserver](https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/standalone/execution_trace_observer.cpp) implemented in PyTorch. The process involves instantiating the observer, registering a callback, and initiating profiling. Although you have the flexibility to collect as many execution traces as desired, for training jobs, profiling a single iteration is advisable for optimal results. To gather these traces, set up the observer and control the start and stop of the profiling. Below is a scripting example for profiling execution traces:

### Collecting Kineto Traces
Next, it's essential to collect Kineto traces, which shed light on the GPU operators within the model. You can collect Kineto traces with torch.profiler.profile. When using torch.profiler.profile, it's important to supply the correct arguments to ensure accurate collection of Kineto traces. Additionally, ensure that prof.step() is called at the end of each iteration. The process includes a warm-up phase, during which the profiler begins tracing but discards the results, followed by an active tracing phase where the profiler traces and records data. Further details can be found in the [PyTorch manual](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs).


### Simultaneous Collection of PyTorch Execution and Kineto Traces

To ensure that traces are linked in the following steps, it's essential to collect PyTorch execution traces and Kineto traces simultaneously during model execution. This approach ensures that the traces align perfectly in terms of timing and events. To achieve this, integrate both the ExecutionTraceObserver and Kineto profiling within the same epoch. Here's an adapted example demonstrating this method:

```python
import torch
from torch.profiler import ExecutionTraceObserver, profile

def trace_handler(prof):
    prof.export_chrome_trace("kineto_trace.json")

def main():
    et = ExecutionTraceObserver()
    et.register_callback("pytorch_et.json")
    et.start()

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=1),
        on_trace_ready=trace_handler
    ) as prof:
        for epoch in ...:
            ...
            if epoch == 6:
                et.stop()
            if epoch == 5:
                et.start()
            ...
            prof.step()

    et.stop()
    et.unregister_callback()
```

Note: to prevent the trace becoming too large, you could just profile three iterations within an epoch. You can adjust the number according to the metrics you want to measure.

## TPU

### Current trace format

1. \<host>.trace.json
2. metric specific trace

### Profiling tensor parallel MaxText workloads

Use the JAX/XLA profiler path exposed by MaxText. For a tensor-parallel training
run, set the mesh dimensions in `ici_mesh_shape`, enable the profiler, and write
the profile to a stable directory outside the repository:

```bash
MAXTEXT_ROOT="${MAXTEXT_ROOT:-./MaxText}"
MODEL_NAME="deepseek-v2-16b"
TP=2
EP=4
DP=1
FSDP=1
PER_DEVICE_BATCH=8
SEQ_LEN=1024
STEPS=10
PROFILE_DIR="/tmp/ccl-bench-maxtext-profile"

python "${MAXTEXT_ROOT}/train.py" \
  MaxText/configs/base.yml \
  model_name="${MODEL_NAME}" \
  ici_mesh_shape="[${DP}, ${TP}, ${FSDP}, ${EP}]" \
  per_device_batch_size="${PER_DEVICE_BATCH}" \
  max_target_length="${SEQ_LEN}" \
  steps="${STEPS}" \
  enable_profiler=true \
  profile_dir="${PROFILE_DIR}"
```

Choose `STEPS` so the profiler captures steady state: at least five warm-up
training steps followed by at least five profiled training steps. The MaxText
workload card should record the collected training iteration count, the TP/EP/DP
mesh values, the TPU type, and `json_tpu` under `metric_source.traces`.

After the run, export the TPU profile as a Chrome trace JSON from the profile
directory. A valid CCL-Bench TPU trace directory should contain the workload card
and one exported file named like:

```text
<hostname-or-worker>.trace.json
```

If TensorBoard is used to inspect the profile, start it on the profile directory,
open the Profile trace viewer, and export/download the trace as JSON:

```bash
tensorboard --logdir "${PROFILE_DIR}" --port 6006
```

Store the exported `*.trace.json` next to the workload card and set
`trace_url` to that JSON file. For example:

```yaml
trace_url: /data/ccl-bench_trace_collection/<workload>/<worker>.trace.json
metric_source:
  traces:
    - json_tpu
```
