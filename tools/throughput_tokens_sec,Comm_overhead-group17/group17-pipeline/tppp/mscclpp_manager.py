import os
import torch
import cupy as cp
import mscclpp.comm as mscclpp_comm
from mscclpp import ProxyService, is_nvls_supported
from mpi4py import MPI
import netifaces as ni
import ipaddress

# 导入你的 Benchmark 中的算子
# 注意：请确保你的 mscclpp_op.py 里能够处理 group=None 的情况
from mscclpp_op import (
    MscclppAllReduce1, MscclppAllReduce2, 
    MscclppAllReduce6
)

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
    # 1. 优先从环境变量获取
    env_ip = os.environ.get("MSCCLPP_COMM_ID_IP")
    if env_ip:
        # 特殊处理：如果是 127.0.0.1，直接返回 loopback 接口
        if env_ip == "127.0.0.1":
            return "lo", env_ip
            
        # 尝试匹配其他接口
        interfaces = ni.interfaces()
        for interface in interfaces:
            addresses = ni.ifaddresses(interface)
            if ni.AF_INET in addresses:
                for addr in addresses[ni.AF_INET]:
                    if addr["addr"] == env_ip:
                        return interface.split(":")[0], env_ip
        # 匹配不到也硬着头皮返回，通常给个 eth0 或 hsn0 名字即可
        return "hsn0", env_ip

    # 2. 自动探测
    interfaces = ni.interfaces()
    for interface in interfaces:
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for addr in addresses[ni.AF_INET]:
                ip_address = addr["addr"]
                if is_valid(ip_address):
                    # === 关键修改 ===
                    safe_interface = interface.split(":")[0]
                    return safe_interface, ip_address
    
    # 3. 回退到 Loopback
    return "lo", "127.0.0.1"

class MscclppManager:
    def __init__(self, rank, world_size, tp_size=1, pp_size=1, base_port=50000):
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        
        if world_size % (tp_size * pp_size) != 0:
            raise ValueError(f"World size ({world_size}) must be divisible by tp_size * pp_size")
            
        self.dp_size = world_size // (tp_size * pp_size)
        
        self.operators = {} 
        self.groups = {} 
        
        self.network_interface, self.my_ip = get_netinterface_info()
        if rank == 0:
            print(f"MSCCLPP using Interface: {self.network_interface}, IP: {self.my_ip}")
        
        # 1. 启动 Proxy
        self.proxy = ProxyService()
        self.proxy.start_proxy()
        
        # 2. 初始化 TP Group
        # Rank 映射: [PP, DP, TP]
        # TP 组: 连续的 tp_size 个 rank
        tp_group_id = rank // tp_size 
        tp_rank = rank % tp_size
        # TP 组通常在节点内，端口偏移 0
        self._init_subgroup("tp", MPI.COMM_WORLD, color=tp_group_id, key=tp_rank, port_offset=0, base_port=base_port)

        # 3. 初始化 DP Group
        # DP 组: 跨步选择
        # PP ID: rank // (dp * tp)
        # TP ID: rank % tp
        # Color: pp_id * tp_size + tp_id (相同 PP 和 TP 的人在一起)
        pp_id = rank // (self.tp_size * self.dp_size)
        tp_id = rank % self.tp_size
        dp_color = pp_id * self.tp_size + tp_id
        
        # DP 组端口偏移 100，避免和 TP 组 Bootstrap 冲突
        self._init_subgroup("dp", MPI.COMM_WORLD, color=dp_color, key=rank, port_offset=100, base_port=base_port)

        MPI.COMM_WORLD.barrier()
        # print(f"[Rank {rank}] Manager Init Done. TP={tp_size}, DP={self.dp_size}")

    def _init_subgroup(self, group_name, parent_comm, color, key, port_offset, base_port):
        """
        使用 MPI_Comm_split 创建子组，并初始化 MSCCLPP Group
        """
        # 1. MPI Split 创建子通信域
        sub_comm = parent_comm.Split(color=color, key=key)
        sub_rank = sub_comm.Get_rank()
        sub_size = sub_comm.Get_size()
        
        # 2. 只有当组大小 > 1 时才需要通信组
        if sub_size > 1:
            # 3. 交换 Root IP (在子组内广播)
            root_ip = sub_comm.bcast(self.my_ip, root=0)
            
            # === 关键修复：避免端口冲突 ===
            # 旧逻辑: port = base_port + port_offset
            # 新逻辑: port = base_port + port_offset + (color * 100)
            # 这样 TP组0 用 50000, TP组1 用 50100，互不干扰！
            
            # 为了防止 DP 组 (offset=100) 和 TP 组 (offset=0) 在 color 较大时重叠
            # 我们给 group_name 也加一个大的偏移权重
            group_type_offset = 0 if group_name == "tp" else 10000
            
            final_port = base_port + group_type_offset + (color * 100)
            
            ifIpPortTrio = f"{self.network_interface}:{root_ip}:{final_port}"
            
            # 打印调试信息，确切看到它在连哪里
            print(f"[Rank {self.rank}] Init '{group_name}' (Color {color}): {ifIpPortTrio}")
            
            # 4. 初始化 MSCCLPP Group
            mscclpp_group = mscclpp_comm.CommGroup(
                interfaceIpPortTrio=ifIpPortTrio, 
                rank=sub_rank, 
                size=sub_size
            )
            self.groups[group_name] = mscclpp_group
        else:
            self.groups[group_name] = None 

        sub_comm.Free() # 释放 MPI 子通信域

    def run_all_reduce(self, bucket_tensor, group_type="dp"):
        if group_type not in self.groups:
            # 可能是因为 tp_size=1 没有初始化 tp 组，静默返回即可
            return 
        
        group = self.groups[group_type]
        if group is None:
            return 

        # --- 数据转换 ---
        # 必须确保 input tensor 是连续的，否则 cupy 可能会出错
        if not bucket_tensor.is_contiguous():
            bucket_tensor = bucket_tensor.contiguous()

        nelem = bucket_tensor.numel()
        dtype_torch = bucket_tensor.dtype
        
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
            return # 不支持的类型忽略

        key = f"{group_type}_{nelem}_{dtype_torch}"
        
        if key not in self.operators:
            # 使用 Cupy 封装 PyTorch 指针
            # Unsafe wrapper, speed is priority
            cp_tensor = cp.asarray(bucket_tensor)
            
            # 选择算子策略
            # 如果是单节点 NVLink，NVLS (AllReduce6) 是最好的
            # 如果不支持 NVLS，则回退到 AllReduce1 (SmChannel / ProxyChannel)
            if is_nvls_supported():
                 # 这里的 nelem 可能需要对齐，具体看算子实现
                op = MscclppAllReduce6(group, nelem, dtype_cp)
            else:
                op = MscclppAllReduce1(group, cp_tensor)
            
            self.operators[key] = op

        op = self.operators[key]
        
        # --- 流同步 ---
        # 1. 确保 PyTorch 之前的计算完成
        torch_stream = torch.cuda.current_stream()
        
        # 2. 创建一个兼容 MSCCLPP 的流对象
        mscclpp_stream = MscclppStreamCompat(torch_stream.cuda_stream)
        
        # 3. 运行 MSCCLPP Kernel (在当前 PyTorch 流上)
        # 注意：不要新建 cupy stream，否则无法与 PyTorch 同步
        if isinstance(op, MscclppAllReduce6):
             # NVLS 通常需要专门的显存拷贝
            op_mem = op.get_memory() 
            cp_tensor = cp.asarray(bucket_tensor)
            
            # D2D Copy -> Reduce -> D2D Copy
            op_mem[:] = cp_tensor 
            op(mscclpp_stream)
            cp_tensor[:] = op_mem 
        else:
            # 直接在原地修改
            op(mscclpp_stream)

    def __del__(self):
        if hasattr(self, 'proxy'):
            self.proxy.stop_proxy()