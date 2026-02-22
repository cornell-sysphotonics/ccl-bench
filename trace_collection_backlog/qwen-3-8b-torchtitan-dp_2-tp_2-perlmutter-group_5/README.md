This folder contains the metadata for our finetuning experiments with:
- FSDP & Tensor Parallelism on Qwen 3 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated the trace for FSDP + TP on Qwen 3 8B.
Our trace's metadata is stored in the .yaml file in this folder.

Results:
********************************************************************

FSDP & TP on Qwen 3 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- local_batch_size = 2
    - seq_len = 1024
        Straggler Delay:  0.3034388732650684
        Straggler Slowdown:  1.435624185184452

