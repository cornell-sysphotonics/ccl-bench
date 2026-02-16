# torchtitan/components/gin/backend.py
import torch
import torch.distributed as dist
import os
# from . import gin_ext
from .process_group import ProcessGroupGIN

def gin_enabled():
    return os.getenv("COMM_BACKEND", "nccl") == "gin"

def init_process_group_gin(pg: dist.ProcessGroup) -> ProcessGroupGIN:
    from . import gin_ext
    world_size = dist.get_world_size(group=pg)
    rank = dist.get_rank(group=pg)

    # Rank 0 creates unique ID
    if rank == 0:
        uid = gin_ext.get_unique_id()  # CPU ByteTensor
    else:
        uid = torch.empty(16, dtype=torch.uint8, device="cpu")  # sizeof(ncclUniqueId) usually 16

    # Broadcast unique ID to all ranks
    dist.broadcast(uid, src=0, group=pg)

    # Initialize NCCL LSA communicator
    gin_ext.init_lsa(world_size, rank, uid)

    return ProcessGroupGIN(pg)

def finalize_process_group_gin():
    from . import gin_ext
    gin_ext.finalize_lsa()
