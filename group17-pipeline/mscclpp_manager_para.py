# mscclpp_manager.py
import torch
import cupy as cp
import mscclpp.comm as mscclpp_comm
from mscclpp import ProxyService, is_nvls_supported
from mpi4py import MPI
import netifaces as ni
import ipaddress

# 确保这些算子在你的 mscclpp_op.py 中都已定义
from mscclpp_op import (
    MscclppAllReduce1, MscclppAllReduce2, 
    MscclppAllReduce3, MscclppAllReduce4, 
    MscclppAllReduce6
)

# --- 定义兼容类 (放在顶层，方便外部导入) ---
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
        
        # 3. 创建高优先级通信流 (用于 Overlap)
        self.comm_stream = torch.cuda.Stream(priority=-1)

    def run_all_reduce(self, bucket_tensor):
        """
        核心执行逻辑
        bucket_tensor: PyTorch Tensor (来自 DDP Bucket)
        """
        # 1. 获取 Tensor 信息
        nelem = bucket_tensor.numel()
        dtype_torch = bucket_tensor.dtype
        element_size = bucket_tensor.element_size()
        total_size_bytes = nelem * element_size
        
        # 映射 Dtype
        if dtype_torch == torch.float16:
            dtype_cp = cp.float16
        elif dtype_torch == torch.float32:
            dtype_cp = cp.float32
        elif dtype_torch == torch.bfloat16:
            try:
                dtype_cp = cp.bfloat16
            except AttributeError:
                dtype_cp = cp.uint16 
        else:
            raise RuntimeError(f"Unsupported dtype: {dtype_torch}")

        # 2. 生成 Key 用于缓存算子
        key = f"{nelem}_{dtype_torch}"
        
        # 3. 智能算子选择 (Lazy Initialization)
        if key not in self.operators:
            # 将 PyTorch Tensor 转换为 CuPy 视图 (Zero-copy)
            cp_tensor = cp.asarray(bucket_tensor)
            
            # --- [关键修改] 参考 Benchmark 的算法选择逻辑 ---
            # 阈值通常设为 1MB (1048576 bytes)
            # 小包用 AllReduce2 (低延迟)，大包看是否支持 NVLS
            
            if total_size_bytes < 1024 * 1024: 
                # 小包优化：使用 AllReduce2
                # 注意：AllReduce2 通常需要 input 和 output，这里做 In-place，所以传两次 cp_tensor
                op = MscclppAllReduce2(self.group, cp_tensor, cp_tensor)
                
            elif is_nvls_supported():
                # 大包 + NVLS：使用 AllReduce6 (最高带宽)
                op = MscclppAllReduce6(self.group, nelem, dtype_cp)
                
            else:
                # 大包 + 无 NVLS：回退到 AllReduce1
                op = MscclppAllReduce1(self.group, cp_tensor)
                
            self.operators[key] = op

        op = self.operators[key]
        
        # 4. 执行 (并行流 Overlap 模式)
        
        compute_stream = torch.cuda.current_stream()
        
        # B. 依赖注入：通信必须等计算流生产出梯度
        self.comm_stream.wait_stream(compute_stream)
        
        # C. 切换到通信流执行
        with torch.cuda.stream(self.comm_stream):
            
            # 准备兼容对象
            cp_stream = cp.cuda.ExternalStream(self.comm_stream.cuda_stream)
            mscclpp_stream_obj = MscclppStreamCompat(self.comm_stream.cuda_stream)
            
            with cp_stream:
                if isinstance(op, MscclppAllReduce6):
                    # AllReduce6 特殊处理：Copy In -> Kernel -> Copy Out
                    op_mem = op.get_memory() 
                    cp_tensor = cp.asarray(bucket_tensor) 
                    
                    # 所有的 Copy 和 Kernel Launch 都在 comm_stream 上排队
                    op_mem[:] = cp_tensor  
                    op(mscclpp_stream_obj) 
                    cp_tensor[:] = op_mem  
                else:
                    # AllReduce1 和 AllReduce2 通常直接操作注册好的指针
                    # 假设 bucket 地址没变（DDP通常不变），直接运行
                    op(mscclpp_stream_obj)

        # D. 依赖注入：后续计算必须等通信完成
        compute_stream.wait_stream(self.comm_stream)

    def __del__(self):
        if hasattr(self, 'proxy'):
            self.proxy.stop_proxy()