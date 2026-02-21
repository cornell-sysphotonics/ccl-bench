This folder contains the metadata for our finetuning experiments with:
- FSDP on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS

We skip the first iteration out of 50 iterations total when profiling. 

Refer to run.txt to see how we generated 3 traces total for FSDP on Llama 3.1 8B. 
Our primary trace's metadata is stored in the .yaml file in this folder.
- Batch size of 2 and sequence length of 1024 to stay consistent with the other models we benchmarked

Results:
********************************************************************
FSDP on Llama 3.1 8B with Torchtitan on Perlmutter on 1 node with 4 GPUS
- local_batch_size = 2
    seq_len = 1024:
        Straggler Delay:  0.07647990586007967
        Straggler Slowdown:  1.0828134724359255
    seq_len = 2048: 
        Straggler Delay: 0.1568638554834126
        Straggler Slowdown: 1.186048073616095

- local_batch_size = 4
    seq_len = 1024:
        Straggler Delay:  0.10109734351034688
        Straggler Slowdown:  1.1124675100028594
    # not enough memory for local_batch_size = 4 and seq_len = 2048