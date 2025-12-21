```

# alloc nodes
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999
# ssh nidxxxxxx

# run training
./run_trace_1n4g.sh

# wait until training finished
# then do Ctrl+C twice and wait for trace generation
```

