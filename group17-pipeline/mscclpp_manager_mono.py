# mscclpp_manager.py
import torch
import cupy as cp
import mscclpp.comm as mscclpp_comm
from mscclpp import ProxyService, is_nvls_supported
from mpi4py import MPI
import netifaces as ni
import ipaddress

# 导入你的 Benchmark 中的算子
from mscclpp_op import (
    MscclppAllReduce1, MscclppAllReduce2, 
    MscclppAllReduce3, MscclppAllReduce4, 
    MscclppAllReduce6
)

# --- 核心修复：定义兼容类 ---
class MscclppStreamCompat:
    """
    一个简单的包装类，用于解决 mscclpp 对流对象的属性检查问题。
    mscclpp/utils.py 期望对象有一个 .cuda_stream 属性（int指针）。
    """
    def __init__(self, stream_ptr):
        self.cuda_stream = stream_ptr 

def is_valid(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return not (ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast)
    except ValueError:
        return False

def get_netinterface_info():
    interfaces = ni.interfaces()
    for interface in interfaces:
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for addr in addresses[ni.AF_INET]:
                ip_address = addr["addr"]
                if is_valid(ip_address):
                    return interface, ip_address
    raise RuntimeError("No valid network interface found")

class MscclppManager:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.operators = {} # 缓存编译好的算子
        
        # 1. 初始化通信组 (Bootstrap)
        network_interface, my_ip = get_netinterface_info()
        root_ip = MPI.COMM_WORLD.bcast(my_ip, root=0)
        ifIpPortTrio = f"{network_interface}:{root_ip}:50000"
        
        print(f"[Rank {rank}] Initializing MscclppGroup with {ifIpPortTrio}")
        
        self.group = mscclpp_comm.CommGroup(
            interfaceIpPortTrio=ifIpPortTrio, 
            rank=self.rank, 
            size=self.world_size
        )
        
        # 2. 启动代理服务
        self.proxy = ProxyService()
        self.proxy.start_proxy()
        MPI.COMM_WORLD.barrier() # 等待所有节点 Proxy 启动

    def run_all_reduce(self, bucket_tensor):
        """
        核心执行逻辑
        bucket_tensor: PyTorch Tensor (来自 DDP Bucket)
        """
        # 1. 获取 Tensor 信息
        nelem = bucket_tensor.numel()
        dtype_torch = bucket_tensor.dtype
        
        # --- 修复点：映射 Dtype (增加 bfloat16 支持) ---
        if dtype_torch == torch.float16:
            dtype_cp = cp.float16
        elif dtype_torch == torch.float32:
            dtype_cp = cp.float32
        elif dtype_torch == torch.bfloat16:
            # 尝试使用 cupy 的 bfloat16
            try:
                dtype_cp = cp.bfloat16
            except AttributeError:
                # 如果当前 CuPy 版本过低不支持 bfloat16，
                # 对于纯通信任务，可以将其视为 uint16 处理，
                # 但需要确保 mscclpp 算子内部不进行 float 数值计算
                dtype_cp = cp.uint16 
        else:
            raise RuntimeError(f"Unsupported dtype: {dtype_torch}")

        # 2. 生成 Key 用于缓存算子
        key = f"{nelem}_{dtype_torch}"
        
        # 3. 如果是第一次遇到这个 Bucket Size，初始化算子
        if key not in self.operators:
            # 将 PyTorch Tensor 转换为 CuPy 视图 (Zero-copy)
            cp_tensor = cp.asarray(bucket_tensor)
            
            if is_nvls_supported():
                # NVLS (AllReduce6) 通常只需要 size 和 dtype 即可初始化
                # 假设 MscclppAllReduce6 在初始化时分配共享内存
                op = MscclppAllReduce6(self.group, nelem, dtype_cp)
                self.operators[key] = op
            else:
                # 简单实现：使用 SmSimple (AllReduce1)
                # 这里的 input 只是为了拿 buffer 地址注册，实际运行可以变
                op = MscclppAllReduce1(self.group, cp_tensor)
                self.operators[key] = op

        op = self.operators[key]
        
        # 4. 执行 (修复了 Stream 处理)
        
        # A. 获取当前 PyTorch Stream (这是 DDP 正在使用的流)
        torch_stream = torch.cuda.current_stream()
        
        # B. 创建 CuPy 上下文流 (用于 copy 操作排队)
        cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
        
        # C. 创建 MSCCLPP 兼容对象 (用于 kernel launch)
        # 这修复了 AttributeError: 'ExternalStream' object has no attribute 'cuda_stream'
        mscclpp_stream = MscclppStreamCompat(torch_stream.cuda_stream)
        
        # D. 在当前流上下文中执行
        with cp_stream:
            if isinstance(op, MscclppAllReduce6):
                # AllReduce6 (NVLS) 通常需要显式的 Copy In/Out 到其内部的 registered buffer
                op_mem = op.get_memory() 
                cp_tensor = cp.asarray(bucket_tensor) # Zero-copy view
                
                # Device to Device copy (async)
                op_mem[:] = cp_tensor 
                
                # Launch Kernel (传入兼容对象)
                op(mscclpp_stream)
                
                # Device to Device copy (async)
                cp_tensor[:] = op_mem 
            else:
                # In-place 运行 (传入兼容对象)
                # 假设 AllReduce1 直接操作传入时的指针地址
                # 如果 Bucket 地址变了，这里可能需要 update_buffer 之类的操作
                # 但 DDP Bucket 每一轮迭代地址通常是不变的
                op(mscclpp_stream)

    def __del__(self):
        if hasattr(self, 'proxy'):
            self.proxy.stop_proxy()