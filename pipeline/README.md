## Useful metrics

1. mfu
group 1: not sure about the mfu calculation methods
group 4: 
```
./scripts/get_mfu_group4.sh
```


## Broken
1. estimated_bandwidth: memory bw
group 4: inference workloads on TPU
```
./scripts/get_estimated_bandwidth_group4.sh ./trace_collection/Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4
```