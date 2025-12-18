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

# --- 兼容类 ---
class MscclppStreamCompat:
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
    def __init__(self, rank, world_size, tp_size=1, pp_size=1, base_port=50000):
        """
        Args:
            rank: Global rank
            world_size: Global world size
            tp_size: Tensor Parallel size
            pp_size: Pipeline Parallel size
            base_port: 起始端口号，不同组会在此基础上偏移以避免冲突
        """
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        
        if world_size % (tp_size * pp_size) != 0:
            raise ValueError(f"World size ({world_size}) must be divisible by tp_size * pp_size ({tp_size * pp_size})")
            
        self.dp_size = world_size // (tp_size * pp_size)
        
        # 缓存编译好的算子，Key 需要包含 group 信息
        self.operators = {} 
        self.groups = {} # 存储 'dp', 'tp', 'world' 等不同的 CommGroup
        
        # 获取物理接口信息
        self.network_interface, self.my_ip = get_netinterface_info()
        
        # 1. 启动全局代理服务 (Proxy 只需要一个，服务于所有 Group)
        self.proxy = ProxyService()
        self.proxy.start_proxy()
        
        # 2. 初始化不同的通信组
        # 假设常见的 Megatron-LM 3D 并行 Rank 映射:
        # 顺序通常是: [PP, DP, TP] -> TP 是最内层连续的 Rank
        
        # --- 初始化 TP Group (Tensor Parallel) ---
        # 相同 PP 和 DP index，不同 TP index 的 rank 组成一个 TP 组
        # 在最简单的设定下，连续的 tp_size 个 rank 是一个 TP 组
        tp_group_id = rank // tp_size 
        tp_rank = rank % tp_size
        self._init_subgroup("tp", MPI.COMM_WORLD, color=tp_group_id, key=tp_rank, port_offset=0, base_port=base_port)

        # --- 初始化 DP Group (Data Parallel) ---
        # 相同 PP 和 TP index，不同 DP index 的 rank 组成一个 DP 组
        # DP 组通常是跨步的。例如 8卡, TP=2. DP组0: [0, 2, 4, 6], DP组1: [1, 3, 5, 7]
        # 计算逻辑:
        # my_pp_idx = rank // (tp_size * dp_size)
        # my_dp_idx = (rank // tp_size) % dp_size
        # my_tp_idx = rank % tp_size
        
        # DP 组由拥有相同 (PP_ID, TP_ID) 的人组成
        pp_id = rank // (self.tp_size * self.dp_size)
        tp_id = rank % self.tp_size
        # 这里的 color 决定了谁在一起。PP 和 TP 相同的人在一起。
        dp_color = pp_id * self.tp_size + tp_id
        # key 决定组内 rank 顺序，用 rank 自身即可 (或者 my_dp_idx)
        self._init_subgroup("dp", MPI.COMM_WORLD, color=dp_color, key=rank, port_offset=100, base_port=base_port)

        # 等待所有组初始化完成
        MPI.COMM_WORLD.barrier()
        print(f"[Rank {rank}] Initialized. TP={tp_size}, DP={self.dp_size}, PP={pp_size}")

    def _init_subgroup(self, group_name, parent_comm, color, key, port_offset, base_port):
        """
        使用 MPI_Comm_split 创建子组，并初始化 MSCCLPP Group
        """
        # 1. MPI Split 创建子通信域
        sub_comm = parent_comm.Split(color=color, key=key)
        sub_rank = sub_comm.Get_rank()
        sub_size = sub_comm.Get_size()
        
        # 2. 只有当组大小 > 1 时才需要通信组，单卡不需要 AllReduce
        if sub_size > 1:
            # 3. 交换 Root IP (在子组内广播)
            root_ip = sub_comm.bcast(self.my_ip, root=0)
            
            # 使用不同的端口避免 Bootstrap 混淆
            port = base_port + port_offset
            ifIpPortTrio = f"{self.network_interface}:{root_ip}:{port}"
            
            # 4. 初始化 MSCCLPP Group
            mscclpp_group = mscclpp_comm.CommGroup(
                interfaceIpPortTrio=ifIpPortTrio, 
                rank=sub_rank, 
                size=sub_size
            )
            self.groups[group_name] = mscclpp_group
            # 记录当前组内的 rank，方便后续逻辑判断
            print(f"[Rank {self.rank}] Init Group '{group_name}': LocalRank={sub_rank}/{sub_size}")
        else:
            self.groups[group_name] = None # 单卡不需要通信

        sub_comm.Free() # 释放 MPI 子通信域

    def run_all_reduce(self, bucket_tensor, group_type="dp"):
        """
        Args:
            bucket_tensor: PyTorch Tensor
            group_type: "dp" (用于梯度聚合) 或 "tp" (用于层内张量并行)
        """
        if group_type not in self.groups:
            raise ValueError(f"Group type '{group_type}' not initialized.")
        
        group = self.groups[group_type]
        
        # 如果该维度的并行度为1 (例如 TP=1)，则直接返回，无需通信
        if group is None:
            return 

        # 1. 获取 Tensor 信息
        nelem = bucket_tensor.numel()
        dtype_torch = bucket_tensor.dtype
        
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

        # 2. 生成 Key (加入 group_type 以隔离不同组的算子)
        key = f"{group_type}_{nelem}_{dtype_torch}"
        
        # 3. 初始化或获取算子
        if key not in self.operators:
            cp_tensor = cp.asarray(bucket_tensor)
            
            if is_nvls_supported():
                # NVLS (AllReduce6)
                op = MscclppAllReduce6(group, nelem, dtype_cp)
                self.operators[key] = op
            else:
                # Simple (AllReduce1)
                op = MscclppAllReduce1(group, cp_tensor)
                self.operators[key] = op

        op = self.operators[key]
        
        # 4. 执行
        torch_stream = torch.cuda.current_stream()
        cp_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
        mscclpp_stream = MscclppStreamCompat(torch_stream.cuda_stream)
        
        with cp_stream:
            if isinstance(op, MscclppAllReduce6):
                op_mem = op.get_memory() 
                cp_tensor = cp.asarray(bucket_tensor)
                op_mem[:] = cp_tensor 
                op(mscclpp_stream)
                cp_tensor[:] = op_mem 
            else:
                op(mscclpp_stream)

    def __del__(self):
        if hasattr(self, 'proxy'):
            self.proxy.stop_proxy()