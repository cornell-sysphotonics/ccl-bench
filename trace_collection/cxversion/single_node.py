import os
import sys
import argparse
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json

# --- 配置常量 ---
# 模型和数据路径 (请根据您的环境调整)
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-v1" # 使用 wikitext-103-v1 作为配置
MAX_LENGTH = 256
#GLOBAL_BATCH_SIZE = 2 # 由 DeepSpeed config 和 per_device_train_batch_size 决定

def parse_args():
    parser = argparse.ArgumentParser(description="Parse all arguments for DeepSpeed/Trainer.")
    
    # --- A. 添加所有的 TrainingArguments 参数 (仅用于捕获值) ---
    # 警告：必须确保这里列出的所有参数，其名称和类型与 TrainingArguments 完全一致
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output_dir")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    
    # --- B. 添加自定义脚本参数 ---
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--max_train_samples", type=int, default=500)
    
    # DeepSpeed 自动注入的参数也需要添加，否则会成为未知参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank, typically injected by DeepSpeed.")

    # 2. 解析所有参数
    args, unknown_args = parser.parse_known_args()
    
    return args

def main():
    args = parse_args()
    
    # 1. 初始化和日志
    if args.local_rank != -1:
        # ⚠️ 在 DeepSpeed/DDP 模式下，必须在所有进程上设置 device
        torch.cuda.set_device(args.local_rank)
    
    transformers.logging.set_verbosity_info()
    if args.local_rank == 0:
        print(f"DeepSpeed Config: {args.deepspeed}")
        print(f"Model: {args.model_name_or_path}")

    # 2. 加载 Tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32, # Llama 推荐 bf16 或 fp16
        use_cache=False
    )
    
    # 3. 数据加载和预处理
    
    # 假设使用 wikitext-103-v1 配置
    raw_datasets = load_dataset(args.dataset_name, DATASET_CONFIG, split="train")
    
    # 限制样本数量
    if args.max_train_samples is not None:
        raw_datasets = raw_datasets.select(range(args.max_train_samples))

    def tokenize_function(examples):
        # 简单地对文本进行分词
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_LENGTH
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Running tokenizer on dataset"
    )

    # 4. 实例化 TrainingArguments
    # 关键：从 CLI 参数重新创建 TrainingArguments 实例，以确保 DeepSpeed 配置被正确传递
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        do_train=args.do_train,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        # 其他从命令行传入的参数，如 logging_steps, save_strategy 等，都已自动被包含
        
        # 调整 Trainer 默认值以匹配 DeepSpeed 零冗余目标
        optim="sgd", # 使用 adamw_hf 或 deepspeed_adam
        save_strategy="no",
        logging_steps=10
    )

    # 5. 实例化 Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. 开始训练
    if training_args.do_train:
        if args.local_rank <= 0:
            print("Starting DeepSpeed/Trainer training...")
        # ⚠️ NVTX 标记的替代方案：如果要进行 nsys 深度剖析，必须使用 srun 包装 nsys profile
        trainer.train()
        if args.local_rank <= 0:
            print("Training finished.")

if __name__ == "__main__":
    main()