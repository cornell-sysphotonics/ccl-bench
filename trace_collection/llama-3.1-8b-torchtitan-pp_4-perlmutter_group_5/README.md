This folder contains the metadata for our finetuning experiments with:
- Pipeline Parallelism on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated the trace for PP on Llama 3.1 8B. 
Our trace's metadata is stored in the .yaml file in this folder.

Results:
********************************************************************
Pipeline Parallelism on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- local_batch_size = 4 # has to be greater than or equal to pipeline_parallel_degree
    seq_len = 512:
        Straggler Delay:  0.1439889409951021
        Straggler Slowdown:  1.168209206505448
    # not enough memory for higher sequence lengths