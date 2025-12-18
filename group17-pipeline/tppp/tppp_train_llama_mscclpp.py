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
from mscclpp_manager import MscclppManager, MscclppStreamCompat

# --- Profiler 相关模块 ---
from torch.profiler import (
    profile, 
    record_function, 
    ProfilerActivity, 
    schedule, 
    ExecutionTraceObserver
)

# --- 1. 参数定义与解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="MSCCLPP Training with 3D Parallelism Setup")
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
    
    # 初始化全局通信域 (NCCL) - 这是 PyTorch 分布式的基础
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return local_rank

# --- 2. 创建 PyTorch DP 通信组 ---
def create_dp_process_group(rank, world_size, tp_size, pp_size):
    """
    为了让 DDP 只在 Data Parallel 的节点间同步梯度，我们需要创建 DP Group。
    假设 Rank 映射顺序为: [PP, DP, TP] (TP 是最内层，连续的 rank)
    """
    dp_size = world_size // (tp_size * pp_size)
    if world_size % (tp_size * pp_size) != 0:
        raise ValueError(f"World size {world_size} not divisible by tp*pp ({tp_size*pp_size})")

    # 计算当前 rank 所属的 DP 组
    # 所有的通信组都需要在所有进程中创建，以保证同步
    dp_group = None
    
    # 遍历所有可能的组合来创建 group
    for pp_idx in range(pp_size):
        for tp_idx in range(tp_size):
            # 找到属于同一个 (PP, TP) 组合的所有 DP 节点
            # 这些节点需要进行梯度同步
            ranks_in_group = []
            for dp_idx in range(dp_size):
                # 假设 rank = pp * (dp * tp) + dp * tp + tp (Megatron 风格)
                # 或者 rank = pp * (dp * tp) + dp * (tp) + tp ?
                # 让我们使用上一轮对话中约定的简单映射:
                # Rank = global index.
                # 简单的步长切分:
                # 拥有相同 PP_ID 和 TP_ID 的 rank 组成一个 DP 组
                # 假设全量 Ranks 排列: [Rank 0, Rank 1, ... Rank N]
                # TP 组: [0, 1] (如果是 TP=2)
                # DP 组通常是跨步的: [0, 2, 4, 6] (如果是 TP=2)
                
                # 计算全局 Rank:
                # global_rank = pp_idx * (dp_size * tp_size) + dp_idx * tp_size + tp_idx
                # 这种映射下，DP 组的 ranks 是跨步为 tp_size 的
                
                r = pp_idx * (dp_size * tp_size) + dp_idx * tp_size + tp_idx
                ranks_in_group.append(r)
            
            # 创建组
            group = dist.new_group(ranks=ranks_in_group, backend="nccl")
            
            # 如果当前进程在这个组里，保存下来作为我的 dp_group
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

# --- Hook 修改：指定 Group Type ---
def mscclpp_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    manager = state
    tensor = bucket.buffer()
    
    # [修改点] DDP 的 Hook 对应的是 Data Parallel 通信
    # 所以这里必须指定 group_type="dp"
    manager.run_all_reduce(tensor, group_type="dp")
    
    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut

def main():
    # 1. 获取 MPI Rank (作为物理 Rank) 和 参数
    args = parse_args()
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    
    # 2. 基础 Setup
    local_rank = setup(rank, world_size)
    
    # 3. 创建 PyTorch DDP 专用的 Process Group
    # 如果 tp=1, pp=1，dp_group 实际上就是全局 group，但也建议通过 new_group 显式创建
    torch_dp_group = create_dp_process_group(rank, world_size, args.tp_size, args.pp_size)
    
    # 4. 初始化 MSCCLPP Manager (传入 TP/PP 参数)
    print(f"Rank {rank}: Initializing MSCCLPP Manager (TP={args.tp_size}, PP={args.pp_size})...")
    mscclpp_manager = MscclppManager(
        rank=rank, 
        world_size=world_size, 
        tp_size=args.tp_size, 
        pp_size=args.pp_size
    )

    # 5. 模型与 LoRA 加载
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 注意：在没有真正模型并行的代码库中，我们只能简单加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=local_rank # 简单的放在当前卡上
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

    # 6. DDP 封装 [修改点]
    # process_group 必须设置为我们计算出的 dp_group。
    # 否则 DDP 会默认在全局进行 AllReduce，这在 TP/PP 开启时是错误的。
    model = DDP(
        model, 
        device_ids=[local_rank], 
        process_group=torch_dp_group 
    )

    # 7. 注册 MSCCLPP Hook
    model.register_comm_hook(state=mscclpp_manager, hook=mscclpp_hook)
    
    train_data = get_c4_dataset(tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=default_data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ================= Trace 收集 =================
    
    et_filename = f"trace/torch_et_mscclpp_rank{rank}.json"
    et = ExecutionTraceObserver()
    et.register_callback(et_filename)
    
    def trace_handler(p):
        output_file = f"trace/kineto_trace_mscclpp_rank{rank}.json"
        p.export_chrome_trace(output_file)
        if rank == 0:
            print(f"Rank {rank}: Kineto trace exported to {output_file}")

    prof_schedule = schedule(wait=2, warmup=2, active=1, repeat=1)

    print(f"Rank {rank}: Starting profiling loop...")
    
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