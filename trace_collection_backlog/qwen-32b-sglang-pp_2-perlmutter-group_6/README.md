# qwen-32b-sglang-pp_2-perlmutter-group_6

Online serving of Qwen3-32b on 8 A100 GPUs on 2 nodes with TP=4, PP=2.

First allocate 2 nodes on Perlmutter.

Run the following command on node 0.

```
MASTER_ADDR=<master_addr> NODE_RANK=0 ./run.sh
```

Run the following command on node 1.

```
MASTER_ADDR=<master_addr> NODE_RANK=1 ./run.sh
```

Where `<master_addr>` is the hostname of node 0. For example, `nid001020`.