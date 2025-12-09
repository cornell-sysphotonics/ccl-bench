import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import torch.distributed as dist
from peft import get_peft_model, LoraConfig, TaskType
from mscclpp_manager import MscclppManager, MscclppStreamCompat

# --- 新增：导入 Profiler 相关模块 ---
from torch.profiler import (
    profile, 
    record_function, 
    ProfilerActivity, 
    schedule, 
    ExecutionTraceObserver
)

def setup():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    os.environ["MASTER_ADDR"] = "localhost" 
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def get_c4_dataset(tokenizer, max_length=512):
    # ... (保持原样，省略以节省空间) ...
    print("Loading C4 dataset (en, validation subset for speed)...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    dataset = dataset.take(1000)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )

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

# --- 定义 Hook ---
def mscclpp_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    # 注意：使用 LoRA 后，这里的 bucket.buffer() 只包含 LoRA 的梯度
    # 数据量会比全量微调小很多，这对 MSCCLPP 的测试可能意味着通信压力变小了
    manager = state
    tensor = bucket.buffer()
    
    manager.run_all_reduce(tensor)
    
    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut

def main():
    rank, local_rank, world_size = setup()
    
    # ... (模型加载、LoRA配置、DDP封装逻辑保持不变) ...
    # 假设这里已经完成了: mscclpp_manager初始化, model加载, LoRA, DDP封装, optimizer定义
    # 为了节省篇幅，这里省略初始化代码，直接进入 Trace 收集部分
    
    # [模拟初始化代码位置]
    print(f"Rank {rank}: Initializing MSCCLPP Manager...")
    mscclpp_manager = MscclppManager(rank, world_size)
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=local_rank)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, peft_config)
    model = DDP(model, device_ids=[local_rank])
    # 注册 Hook
    def mscclpp_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        manager = state
        tensor = bucket.buffer()
        manager.run_all_reduce(tensor)
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut
    model.register_comm_hook(state=mscclpp_manager, hook=mscclpp_hook)
    
    train_data = get_c4_dataset(tokenizer)
    train_loader = DataLoader(train_data, batch_size=10, collate_fn=default_data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ================= Trace 收集核心代码 =================
    
    # 1. 定义 ET Observer
    # 注意：文件名加上 rank，防止多卡写入冲突
    et_filename = f"trace/torch_et_mscclpp_rank{rank}.json"
    et = ExecutionTraceObserver()
    et.register_callback(et_filename)
    
    # 2. 定义 Kineto Handler
    def trace_handler(p):
        output_file = f"trace/kineto_trace_mscclpp_rank{rank}.json"
        p.export_chrome_trace(output_file)
        if rank == 0:
            print(f"Rank {rank}: Kineto trace exported to {output_file}")

    # 3. 开启 Profiler 上下文
    # Schedule: Wait 2 -> Warmup 2 -> Active 1 (在 Step 4 进行抓取)
    prof_schedule = schedule(wait=2, warmup=2, active=1, repeat=1)

    print(f"Rank {rank}: Starting profiling loop...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True, # 如果想看显存分配情况可开启
        with_stack=True
    ) as prof:
        
        model.train()
        
        for step, batch in enumerate(train_loader):
            if step >= 10: break
            
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            # --- 同步 ET 和 Kineto ---
            # 我们设定的 active 是第 5 个 step (index 4)
            # wait=2 (steps 0,1), warmup=2 (steps 2,3), active=1 (step 4)
            if step == 4:
                if rank == 0: print("Rank 0: Starting Execution Trace Observer...")
                et.start()
                # torch.cuda.cudart().cudaProfilerStart()
            
            # 训练步
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # --- 停止 ET ---
            if step == 4:
                if rank == 0: print("Rank 0: Stopping Execution Trace Observer...")
                et.stop()
                # torch.cuda.cudart().cudaProfilerStop()
            
            # Kineto Step (这一步会自动判断是否进入 Active 状态并记录)
            prof.step()
            
            if rank == 0:
                print(f"Step {step} completed.")

    # 清理
    et.unregister_callback()
    print(f"Rank {rank}: Training finished.")
    destroy_process_group()

if __name__ == "__main__":
    main()