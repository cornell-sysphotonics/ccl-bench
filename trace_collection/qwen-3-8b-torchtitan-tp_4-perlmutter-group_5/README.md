This folder contains the metadata for our finetuning experiments with:
- Tensor Parallelism on Qwen 3 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated 3 traces total for TP on Qwen 3 8B
Our primary trace's metadata is stored in the .yaml file in this folder.
- Batch size of 2 and sequence length of 1024 to stay consistent with the other models we benchmarked

Results:
********************************************************************

Tensor Parallelism on Qwen 3 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- tensor_parallel_degree = 4
- seq_len = 1024

local_batch_size = 1:
    Straggler Delay:  0.8180599421073073
    Straggler Slowdown:  5.496315718387836

local_batch_size = 2:
    Straggler Delay:  0.1970835109562599
    Straggler Slowdown:  1.245459538626468

local_batch_size = 4:
    Straggler Delay:  0.030754125504352005
    Straggler Slowdown:  1.031729952444064