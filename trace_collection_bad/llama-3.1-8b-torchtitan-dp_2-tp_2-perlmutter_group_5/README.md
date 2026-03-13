This folder contains the metadata for our finetuning experiments with:
- FSDP & Tensor Parallelism on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated the traces for FSDP + TP on Llama 3.1 8B. 
Our trace's metadata is stored in the .yaml file in this folder.

Results:
********************************************************************
FSDP & TP on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- local_batch_size = 2
- seq_len = 1024

    Straggler Delay:  0.10903784368806896
    Straggler Slowdown:  1.1223821269125758
