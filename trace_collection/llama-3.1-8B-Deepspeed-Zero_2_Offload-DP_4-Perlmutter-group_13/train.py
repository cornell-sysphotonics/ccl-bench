import os
import torch
import deepspeed
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.1-8B" 

OUTPUT_DIR = "./output_llama_zero2"
MAX_LENGTH = 64

def train():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[Rank {local_rank}] Loading tokenizer from {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[Rank {local_rank}] Loading dataset...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LENGTH
        )

    small_dataset = dataset.select(range(1000)) 
    tokenized_dataset = small_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"[Rank {local_rank}] Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=1,
        max_steps=20,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        optim="sgd", 
        deepspeed="ds_config_zero2.json"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"[Rank {local_rank}] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print(f"[Rank {local_rank}] Starting training...")
    trainer.train()
    print(f"[Rank {local_rank}] Training finished.")

if __name__ == "__main__":
    train()