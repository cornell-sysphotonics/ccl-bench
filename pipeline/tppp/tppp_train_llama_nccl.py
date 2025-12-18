import os
import argparse  # 新增
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import torch.distributed as dist
from peft import get_peft_model, LoraConfig, TaskType

# --- Profiler 相关模块 ---
from torch.profiler import (
    profile, 
    record_function, 
    ProfilerActivity, 
    schedule, 
    ExecutionTraceObserver
)

# --- 1. 参数定义与解析 (与 MSCCLPP 脚本保持一致) ---
def parse_args():
    parser = argparse.ArgumentParser(description="NCCL Baseline Training with 3D Parallelism Setup")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size per GPU")
    return parser.parse_args()

def setup(rank, world_size):
    # 配置常规环境变量
    os.environ["MASTER_ADDR"] = "localhost" 
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    # 初始化全局通信域 (NCCL)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return local_rank

# --- 2. 创建 PyTorch DP 通信组 (与 MSCCLPP 脚本逻辑完全一致) ---
def create_dp_process_group(rank, world_size, tp_size, pp_size):
    """
    即使是 Baseline，我们也需要模拟 3D 并行的拓扑结构。
    如果 TP > 1，DDP 不应该在所有 GPU 上做 AllReduce，而只应该在 DP 组内做。
    """
    dp_size = world_size // (tp_size * pp_size)
    if world_size % (tp_size * pp_size) != 0:
        raise ValueError(f"World size {world_size} not divisible by tp*pp ({tp_size*pp_size})")

    dp_group = None
    
    # 遍历所有可能的组合来创建 group
    for pp_idx in range(pp_size):
        for tp_idx in range(tp_size):
            ranks_in_group = []
            for dp_idx in range(dp_size):
                # Rank 计算逻辑需与 MSCCLPP 脚本保持严格一致
                r = pp_idx * (dp_size * tp_size) + dp_idx * tp_size + tp_idx
                ranks_in_group.append(r)
            
            # 创建 NCCL 组
            group = dist.new_group(ranks=ranks_in_group, backend="nccl")
            
            if rank in ranks_in_group:
                dp_group = group
                print(f"[Rank {rank}] Assigned to PyTorch DP Group with ranks: {ranks_in_group}")

    return dp_group

def get_c4_dataset(tokenizer, max_length=512):
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

def main():
    # 1. 获取 MPI Rank 和 参数
    args = parse_args()
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    
    # 2. 基础 Setup
    local_rank = setup(rank, world_size)
    
    # 3. 创建 PyTorch DDP 专用的 Process Group
    torch_dp_group = create_dp_process_group(rank, world_size, args.tp_size, args.pp_size)
    
    # 4. 模型与 LoRA 加载 (保持一致)
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 注意：Baseline 同样只进行简单加载，不应用 parallelize_module，以保证计算图与 MSCCLPP 脚本一致
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=local_rank 
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1, 
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 5. DDP 封装 [关键点]
    # 我们不注册 Hook，让它使用默认的 NCCL 后端。
    # 但我们必须传入 process_group=torch_dp_group。
    # 这样可以确保 Baseline 和 MSCCLPP 跑在完全相同的通信环路（Ring/Tree）大小上。
    model = DDP(
        model, 
        device_ids=[local_rank], 
        process_group=torch_dp_group 
    )
    
    train_data = get_c4_dataset(tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=default_data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ================= Trace 收集 =================
    # 文件名加上 _nccl 以示区别
    et_filename = f"trace/torch_et_nccl_rank{rank}.json"
    et = ExecutionTraceObserver()
    et.register_callback(et_filename)
    
    def trace_handler(p):
        output_file = f"trace/kineto_trace_nccl_rank{rank}.json"
        p.export_chrome_trace(output_file)
        if rank == 0:
            print(f"Rank {rank}: Kineto trace exported to {output_file}")

    prof_schedule = schedule(wait=2, warmup=2, active=1, repeat=1)

    print(f"Rank {rank}: Starting profiling loop (NCCL Baseline)...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        model.train()
        
        for step, batch in enumerate(train_loader):
            if step >= 10: break
            
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            if step == 4:
                if rank == 0: print("Rank 0: Starting Execution Trace Observer...")
                et.start()
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if step == 4:
                if rank == 0: print("Rank 0: Stopping Execution Trace Observer...")
                et.stop()
            
            prof.step()
            
            if rank == 0:
                print(f"Step {step} completed.")

    et.unregister_callback()
    print(f"Rank {rank}: Training finished.")
    destroy_process_group()

if __name__ == "__main__":
    main()