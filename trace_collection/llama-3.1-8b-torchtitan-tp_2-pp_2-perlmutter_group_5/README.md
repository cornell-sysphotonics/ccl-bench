This folder contains the metadata for our finetuning experiments with:
- Tensor Parallelism & Pipeline Parallelism on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated the traces for TP + PP on Llama 3.1 8B. 
Our trace's metadata is stored in the .yaml file in this folder.

Results:
********************************************************************
TP & PP on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS:
- local_batch_size = 4
- seq_len = 512
    Straggler Delay:  0.22174988362164813
    Straggler Slowdown:  1.28493395497784