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
    DataCollatorForLanguageModeling
)
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.partition_parameters import Init
from datasets import load_dataset

original_set_device = get_accelerator().set_device
def patched_set_device(device_index):
    original_set_device(0)
get_accelerator().set_device = patched_set_device

original_device_name = get_accelerator().device_name
def patched_device_name(device_index=None):
    return "cuda:0"
get_accelerator().device_name = patched_device_name

original_configure = DeepSpeedEngine._configure_distributed_model
def patched_configure(self, model):
    self.device = torch.device("cuda:0")
    return original_configure(self, model)
DeepSpeedEngine._configure_distributed_model = patched_configure

original_convert = Init._convert_to_zero_parameters
def patched_convert(self, param_list):
    target_device = torch.device("cuda:0")
    self.local_device = target_device
    self.device = target_device
    self.remote_device = target_device
    return original_convert(self, param_list)
Init._convert_to_zero_parameters = patched_convert

# =================================================================

MODEL_PATH = "/pscratch/sd/r/rb945/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DATA_CACHE_DIR = os.path.expandvars("$PSCRATCH/huggingface_datasets")
OUTPUT_DIR = "./output_llama_trace"
MAX_LENGTH = 512

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
    
    if local_rank != -1:
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            print(f"[Rank {local_rank}] Dist initialized. Physical isolation enabled (Mapped to cuda:0).")

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
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
    # only use 1000 data for test
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
        max_steps=10,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        optim="sgd",
        deepspeed="ds_config.json"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    print("Initializing Custom Trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting training (Native Transformers)...")
    trainer.train()
    print("Training finished.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(0)

if __name__ == "__main__":
    train()