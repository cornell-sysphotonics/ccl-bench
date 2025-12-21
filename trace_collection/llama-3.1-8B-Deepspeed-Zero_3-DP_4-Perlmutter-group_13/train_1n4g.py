import sys
import os
import torch
import torch.distributed as dist
import deepspeed
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset

MODEL_PATH = "/pscratch/sd/r/rb945/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DATA_CACHE_DIR = os.path.expandvars("$PSCRATCH/huggingface_datasets")
OUTPUT_DIR = "./output_llama_trace"
MAX_LENGTH = 512

class NvtxCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.nvtx.range_push(f"Train Step {state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.nvtx.range_pop()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"].to(model.device)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

def train():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading wikitext dataset from {DATA_CACHE_DIR}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=DATA_CACHE_DIR)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LENGTH
        )

    print("Tokenizing dataset...")
    small_dataset = dataset.select(range(1000)) 
    tokenized_dataset = small_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.bfloat16,
        use_cache=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=1,
        max_steps=50,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        optim="sgd",
        deepspeed="ds_config.json"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Initializing Custom Trainer with NVTX Callback...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[NvtxCallback()] 
    )

    print("Starting training (Native Transformers with DeepSpeed)...")
    trainer.train()
    print("Training finished.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(0)

if __name__ == "__main__":
    train()