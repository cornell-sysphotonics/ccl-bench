# Trace generation

Trace definition: Arjun, Abhishek

Trace collection method: Eric

## Current trace format
1. torch_et_\<rank>.json
2. kineto_trace_\<rank>.json
3. nsys_\<rank>.nsys-rep
4. metric specific trace

## Trace collection method (from [Chakra Execution Trace Collection](https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces) guide)

This section focuses on simultaneous collection methods for PyTorch execution traces and Kineto traces.
### Collecting PyTorch Execution Traces
You can collect PyTorch execution traces from a PyTorch model's execution. This is achieved by using the [ExecutionTraceObserver](https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/standalone/execution_trace_observer.cpp) implemented in PyTorch. The process involves instantiating the observer, registering a callback, and initiating profiling. Although you have the flexibility to collect as many execution traces as desired, for training jobs, profiling a single iteration is advisable for optimal results. To gather these traces, set up the observer and control the start and stop of the profiling. Below is a scripting example for profiling execution traces:

```python
from torch.profiler import _ExperimentalConfig, ExecutionTraceObserver

et = ExecutionTraceObserver()
et.register_callback("pytorch_et.json")
et.start()
...
et.stop()
et.unregister_callback()
```
An implementation example of the ExecutionTraceObserver can be found in [the param benchmark code](https://github.com/facebookresearch/param/blob/main/train/compute/python/pytorch/run_benchmark.py), which illustrates how to collect execution traces from PyTorch.

### Collecting Kineto Traces
Next, it's essential to collect Kineto traces, which shed light on the GPU operators within the model. You can collect Kineto traces with torch.profiler.profile. When using torch.profiler.profile, it's important to supply the correct arguments to ensure accurate collection of Kineto traces. Additionally, ensure that prof.step() is called at the end of each iteration. The process includes a warm-up phase, during which the profiler begins tracing but discards the results, followed by an active tracing phase where the profiler traces and records data. Further details can be found in the [PyTorch manual](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs).

```python
import torch

def trace_handler(prof):
    prof.export_chrome_trace("./kineto_trace.json")

def main():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=1),
        record_shapes=True,
        on_trace_ready=trace_handler,
    ) as prof:
        ...
        prof.step()
```
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