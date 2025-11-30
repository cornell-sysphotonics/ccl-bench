#!/bin/bash

deepspeed --num_gpus=4 1.py \
  --deepspeed ds_config.json \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --per_device_train_batch_size 1 \
  --output_dir output_dir \
  --overwrite_output_dir \
  --bf16 \
  --do_train \
  --max_train_samples 500 \
  --num_train_epochs 1 \
  --dataset_name Salesforce/wikitext

echo "DeepSpeed job submitted to 2 GPUs."