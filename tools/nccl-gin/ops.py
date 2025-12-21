# torchtitan/components/gin/ops.py
import os
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

# import the CUDA extension
from . import gin_ext
from .backend import gin_enabled

def gin_init_if_needed():
    # Do once per process.
    # rank0 makes unique ID, broadcasts via torch.distributed, then init LSA comm.
    if getattr(gin_init_if_needed, "_inited", False):
        return

    world = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        uid = gin_ext.get_unique_id()  # CPU uint8
    else:
        uid = torch.empty((128,), dtype=torch.uint8, device="cpu")  # sizeof(ncclUniqueId) is 128 usually; safe if matches compiled size
    dist.broadcast(uid, src=0)

    gin_ext.init_lsa(world, rank, uid)
    gin_init_if_needed._inited = True

def gin_all_gather_into_tensor(out: torch.Tensor, inp: torch.Tensor, world_size: int, rank:int):
    # out: [world_size * chunk], inp: [chunk]
    gin_ext.all_gather_into_tensor(out, inp)

def gin_all_reduce(tensor, reduce_op=dist.ReduceOp.SUM, group=None):
    # For FSDP-first milestone: fallback is fine
    # You can keep functional collective or dist.all_reduce
    dist.all_reduce(tensor, op=reduce_op, group=group)
    return tensor

def gin_reduce_scatter_tensor(out: torch.Tensor, inp: torch.Tensor, reduce_op=dist.ReduceOp.SUM, group=None):
    """
    LSA-backed reduce_scatter (sum) using gin_ext.
    out: [chunk]
    inp: [world_size * chunk]
    """
    # For now we ignore reduce_op and assume SUM, matching the CUDA kernel.
    gin_ext.reduce_scatter_into_tensor(out, inp)
    return out
