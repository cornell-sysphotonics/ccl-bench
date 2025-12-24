This folder contains the metadata for our finetuning experiments with:
- FSDP on Qwen 3 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated 2 traces total for FSDP on Qwen 3 8B
Our primary trace's metadata is stored in the .yaml file in this folder.
- Batch size of 2 and sequence length of 1024 to stay consistent with the other models we benchmarked

Results:
********************************************************************
FSDP on Qwen 3 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- data_parallel_shard_degree = 4
- seq_len = 1024

local_batch_size = 1: 
    Straggler Delay:  0.19369118092570067
    Straggler Slowdown:  1.2402195986744535

local_batch_size = 2: 
    Straggler Delay: 0.1074101445075549
    Straggler Slowdown: 1.1203353856719516

# not enough memory for higher batch sizes