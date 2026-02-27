# train_llama_nccl.py (Baseline)
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
from peft import get_peft_model, LoraConfig, TaskType

# Profiler imports
from torch.profiler import (
    profile, ProfilerActivity, schedule, ExecutionTraceObserver
)

def setup():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    # Slurm 环境下通常不需要手动设置 MASTER_ADDR，但为了保险：
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
        
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def get_c4_dataset(tokenizer, max_length=512):
    # (保持与原代码完全一致的数据处理逻辑)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    dataset = dataset.take(1000)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    data_list = []
    for i, item in enumerate(dataset):
        tokenized = tokenize_function(item)
        data_list.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0)
        })
        if i >= 100: break 
    return data_list

def main():
    rank, local_rank, world_size = setup()
    
    # 1. 模型加载 (保持完全一致)
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    if rank == 0: print(f"Rank {rank}: Loading Model {model_id} (NCCL Baseline)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=local_rank)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 2. LoRA 配置 (保持完全一致)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=8, lora_alpha=32, lora_dropout=0.1, 
        target_modules=["q_proj", "v_proj"] # 保持一致
    )
    model = get_peft_model(model, peft_config)
    
    # 3. DDP 封装 (无 Hook!)
    model = DDP(model, device_ids=[local_rank])
    
    train_data = get_c4_dataset(tokenizer)
    train_loader = DataLoader(train_data, batch_size=10, collate_fn=default_data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 4. Profiling 设置
    # 文件名区分：加上 _nccl
    et_filename = f"trace/torch_et_nccl_rank{rank}.json"
    et = ExecutionTraceObserver()
    et.register_callback(et_filename)
    
    def trace_handler(p):
        output_file = f"trace/kineto_trace_nccl_rank{rank}.json"
        p.export_chrome_trace(output_file)

    prof_schedule = schedule(wait=2, warmup=2, active=1, repeat=1)

    if rank == 0: print("Rank 0: Starting training loop (NCCL)...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True
    ) as prof:
        
        model.train()
        
        for step, batch in enumerate(train_loader):
            if step >= 10: break
            
            # # === 关键：手动触发 Nsys Capture (Step 4) ===
            if step == 4:
                et.start()
                # torch.cuda.cudart().cudaProfilerStart()
                # if rank == 0: print(">>> Nsys Capture Started")

            batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if step == 4:
                et.stop()
                # # 停止 Nsys Capture
                # torch.cuda.cudart().cudaProfilerStop()
                # if rank == 0: print("<<< Nsys Capture Stopped")

            prof.step()
            if rank == 0: print(f"Step {step} done.")

    et.unregister_callback()
    destroy_process_group()

if __name__ == "__main__":
    main()