This folder contains the metadata for our finetuning experiments with:
- Tensor Parallelism on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated 3 traces total for TP on Llama 3.1 8B. 
Our primary trace's metadata is stored in the .yaml file in this folder.
- Batch size of 2 and sequence length of 1024 to stay consistent with the other models we benchmarked

Results:
********************************************************************
Tensor Parallelism on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- tensor_parallel_degree = 4

- local_batch_size = 2
    seq_len = 1024:
        Straggler Delay:  0.19500364058833317
        Straggler Slowdown:  1.242241642845257
    seq_len = 2048:
        Straggler Delay:  0.03987708183012887
        Straggler Slowdown:  1.0415333089914574

- local_batch_size = 4
    seq_len = 1024:
        Straggler Delay:  0.015796089666565122
        Straggler Slowdown:  1.0160496107571992