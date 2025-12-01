# **Project Proposal: GPU NV-Link Capacity Utilization**

**Project Number:** 16  
**Members:** Elliot Kim (emk255), Grant Rinehimer (gwr47), Evan Leong (epl44), Kevin Tang (qt58)  
**Advisor:** Abhishek Vijaya Kumar

## **Project Description**

* **Background**: During distributed LLM serving using tensor and expert parallelism, the NV-Links connecting the GPUs are utilized in bursts and are idle otherwise.   
* **Objective**: Profile GPU NV-Links usage while serving different models on vlLM and Sglang on various datasets. (E.g, sharegpt, burstgpt, loogle etc). Specifically evaluate at least 1 MoE and 1 regular LLM.   
* **Deliverables**  
  * A timeline of nvlinks utilization \- this should be extracted using nvml library.  
  * A graph for each model-dataset pair  
  * Report the different sources of traffic on nvlinks  
  * A framework to reproduce the results

# **Background and Scope**

## **Serving Frameworks: vLLM and SGLang**

Both vLLM and SGLang are open source frameworks/engines for serving LLMs. 

vLLM uses PagedAttention, which essentially instead of allocating a contiguous chunk in GPU memory for the KV cache for all sequences in a batch (and padding smaller sequences leading to internal, external fragmentation), vLLM uses a virtual memory style architecture where the entire memory for the KV cache is split into fixed size pages of blocks, and the system is able to dynamically lookup these pages in memory, reducing overall memory usage, and consequentially  increasing throughput by allowing for larger batches. 

SGLang uses Radix Trees, which are prefix trees/tries, where sequences that share the same prefix tokens follow the same nodes in the prefix tree, allowing the system to quickly lookup previously computed KV cache, without having to recompute.

With both frameworks being designed to optimize for memory and throughput, but with different techniques, we expect to see similar results between serving engines, but major differences might start to appear when different degrees of parallelism are introduced. 

## **Datasets**

We will use datasets that are chosen to stress distinct NV-Links communication.   
Baseline Chat  
As a baseline workload, we will use subsets of [ShareGPT](https://huggingface.co/datasets/theblackcat102/sharegpt-english) and the [LMSYS Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) style conversation logs. We will sample single and multi-turn conversations that contain short to medium input prompts, giving us a realistic interactive chat setting.

Short, High Concurrency Bursts  
We will use the [BurstGPT](https://github.com/HPMLL/BurstGPT/releases/tag/v1.1) dataset to simulate bursty workloads. However, this dataset only provides timestamps and token counts for each request, not the text content, due to privacy reasons. We can combine these traces with prompts from ShareGPT and LMSYS Chat-1M so that, at each timestamp, we issue a request whose prompt length matches or is very similar to the Request tokens field of the trace. This lets us reproduce realistic burst patterns without needing access to the original user queries.

Infrequent, Long Inputs  
To investigate infrequent but context-heavy queries, we will use [LooGLE](https://huggingface.co/datasets/bigai-nlco/LooGLE), as a source of prompts that contain long documents or passages. We will focus on requests with long inputs and modest generation length in order to stress NV-Links usage that arises from long context attention and KV cache synchronization rather than from high request volume.

Task-Specific Workload Investigation (If possible)  
Finally, we will expand our investigation based on the types of tasks. One domain of tasks is heavy usage of code, whether that is inputting or generating code. We will refer to benchmarks like [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval/viewer/openai_humaneval/test?row=14) and [MBPP](https://huggingface.co/datasets/Muennighoff/mbpp). These tasks typically induce long, structured generations and are a natural setting to study how code-oriented workloads and potential expert specialization in MoE models affect NVLinks utilization. Another domain of tasks is reasoning for math. We will use datasets such as [GSM8K](https://huggingface.co/datasets/openai/gsm8k) to study the effect of math-related tasks on NV-Links usage. However, we will primarily focus on setting up the pipeline using ShareGPT and LooGLE to investigate short/long prompts, then expand.

## **Models**

We seek to evaluate NV-Link usage on one **dense** model and one **Mixture-Of-Experts** model. We study the **Qwen3** family of models. The **Qwen3** family provides state-of-the-art open-source models at a variety of sizes, including both dense and MoE architectures. Since we evaluate on Perlmutter nodes containing 4 NVIDIA A100s (40 GB VRAM), we choose [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) and [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) as our dense and MoE models, respectively. Each of these models has roughly \~64 GB of weights, so using tensor and expert parallelism when applicable should shard the model weights enough to keep the model weights in VRAM over a single node. 

## **Recording Metrics**

**NVML Usage**  
[https://docs.nvidia.com/deploy/nvml-api/group\_\_NvLink.html](https://docs.nvidia.com/deploy/nvml-api/group__NvLink.html)  
[https://docs.nvidia.com/deploy/nvml-api/group\_\_nvmlFieldValueQueries.html](https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueQueries.html)

Above is the API reference guide for NVML

We will mainly be using [nvmlDeviceGetFieldValues](https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueQueries.html#group__nvmlFieldValueQueries_1g0b02941a262ee4327eb82831f91a1bc0) with NVML\_FI\_DEV\_NVLINK\_THROUGHPUT\_\* as field values to get the utilizations.

Some other useful functions:  
nvmlDeviceGetNvLinkCapability \- compare throughput against  
nvmlDeviceGetNvLinkState \- useful at start to check state of nvlink  
nvmlDeviceGetNvLinkInfo \- maybe needed additional info for the nvlink  
nvmlDeviceGetNvLinkRemotePciInfo\_v2 \- check which remote GPUs are connected  
nvmlDeviceGetNvLinkRemoteDeviceType \- check which remote GPUs are connected

# **Implementation Plans**

Our first goal is to develop a script that takes as input a model name, a dataset, and levels of expert and tensor parallelism. Based on this configuration, it will launch inference with the dataset using offline vLLM. It will perform inference on the dataset with varying prompt batch sizes, up to the maximum batch size that can properly fit on the GPU VRAM. It will also launch a script that uses NVML to record the usage on every NVLINK. Nsys profiles (for each batch size) will also be created during inference to track sources of traffic. After all raw data is collected, an additional script will take raw data and a list of desired metrics as input and output graphs and metrics requested. Afterwards, we will expand our benchmarking script to allow for online inference for both vLLM and Sglang, automatically constructing the server and sending requests using the dataset provided. This will allow us to benchmark bursty requests. The final implementation will allow the user to choose a model, dataset, serving framework, and levels of expert/tensor parallelism, and receive back data on NVLINK usage correlated with NCCL operations for varying prompt batch sizes.

**Step 1: Offline Inference with VLLM**  
First, we will set up our benchmarking script for launching [offline inference](https://docs.vllm.ai/en/v0.11.0/serving/offline_inference.html) with vLLM. This script will handle loading the model name from HuggingFace, batching together prompts from the provided dataset, and performing inference with vLLM. The script will run through the dataset multiple times with different prompt batch sizes, up to the batch size at which the KV caches can no longer fit into GPU VRAM. We will detect this through either estimating the max batch size or detecting a drastic spike in TTFT. This part of our script will give us a simple framework for running inference so that we can properly test our monitoring scripts. 

**Step 2: Integrate NVML to monitor NV-Links Usage and add Nsight Systems Profiling**  
We will create a Python (or C++) script using NVML libraries to continuously record NVLink usage. We will also create scripts to launch nsys profiling to collect information on when or how often NCCL operations occur. These scripts will be combined with the offline inference scripts so that all raw data is recorded automatically when inference begins/ends.

**Step 3: Implement “Bursts” of Data Loaders with online vLLM and sglang**  
Will create dataset loaders for ShareGPT, LooGLE that for different workload types. Using BurstGPT’s trace dataset, each loader will select a prompt whose token length matches the Request tokens field and schedule it at a given timestamp. This hopefully helps us to replicate the failure modes they observed. This will also involve expanding our inference scripts to support online inference, so additional logic will be needed to launch and query the inference server.

**Step 4: Run experiments, collect data**  
With the pipeline and datasets in place, we will run systematic experiments to explore how NVLink usage changes across configurations. For each combination of model (dense or MoE) and dataset, we will sweep over batch sizes and tensor or expert parallel degrees that fit within the 4 GPU memory limits. This will produce a grid of runs that can be directly compared.

**Step 5: Analyze trace and package as reproducible framework**  
We will develop scripts that parse NVML logs and Nsight SQLite database and generate graphs. Using Nsight data, we will correlate spikes in NVLink with specific NCCL operations. All of the code will be organized into a simple framework with configuration files and a single entry point that can run experiments end to end.

# **Expected Results**

In our benchmark tests, when testing ShareGPT versus LooGLE on the same model and degree of parallelism, we expect overall less memory utilization for ShareGPT’s smaller prompts due to the shorter sequence length and smaller attention memory.

Across time, we expect ShareGPT and BurstGPT to have more sustained but lower utilization throughout the entire process. Whereas LooGLE may result in small bursts of high usage as the long context of a single prompt is processed. 

We expect MOE NVLink utilization to be more inconsistent and bursty, with cross-GPU communication being heavily dependent on the prompt and the sparseness of experts across GPUs.

Whereas with tensor parallelism, there is a consistent, predetermined amount of activation data that needs to be sent between GPUs, and hence should result in a steadier utilization pattern.

Comparing different degrees of tensor parallelism, we expect that overall memory utilization to slightly increase across 2 to 4 GPUs, but we also expect individual NVLink memory utilization will likely slightly decrease as the degree of tensor parallelism increases. 

# **Expected Challenges** 

At this point in time, we expect several challenges might arise: 

1. BurstGPT only has timestamps and token length, but no prompts, so we will need to control input/output token length of prompts accordingly, find a way to generate these prompts,and find a method to schedule these prompts asynchronously.   
2. NVML get utilization calls have low granularity/frequency with Python, so we may have to consider C++ NVML library or some other method.  
3. We expect that the pipeline for sending prompts and serving models with SGLang will be relatively similar to the pipeline for vLLM but some minute discrepancies between the two frameworks could cause us to have to redesign the pipeline. 

# **Estimated GPU Hours**

Each member of the group has around 15-25 hours of GPU hours allocated. We estimate that we will not need more GPU hours than our current state because we will only need it for running inference. We will first test how long inference takes for a certain batch size for Qwen3 models, and will choose the specific model accordingly.

### **Rough Estimate of GPU Hours**

Based on preliminary speed benchmarks provided by the Qwen3 development team, the slowest model we plan to benchmark (Qwen3-32B) produces \~26 tokens/s on a single GPU with batch size of 1\. Based on the generation length they used in their own benchmarks (2048 tokens), this means 78 seconds per prompt. If we run 5 trials (batches) per 4 different batch sizes, the entire benchmarking run should take approximately 0.5 hours. We only need to run a few trials (batches) per batch size since the NVLINK usage should already begin to show a pattern per batch processed. Of course, we are making some simplifying assumptions that each prompt batch will run in 78 seconds, but we are also not accounting for speedups from tensor or expert parallelism. With \~80 hours of GPU time available from across the team, this should be more than enough to run a variety of tests. We can also easily make adjustments to the scope and size of our tests (like number of trials or batch sizes) to fit within our budget while not losing much scope to our results.

# **Resources**

Offline inference: [https://docs.vllm.ai/en/v0.11.0/serving/offline\_inference.html](https://docs.vllm.ai/en/v0.11.0/serving/offline_inference.html)  
Dense Qwen3: [https://huggingface.co/Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)  
MoE Qwen3: [https://huggingface.co/Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)  
Qwen3 Speed Benchmarks: [https://qwen.readthedocs.io/en/latest/getting\_started/speed\_benchmark.html](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)  
