# torchtitan/components/gin/process_group.py
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from . import ops

class ProcessGroupGIN(ProcessGroup):
    def __init__(self, pg: ProcessGroup):
        super().__init__()
        self._pg = pg

    def getBackendName(self):
        return "gin"

    def allreduce(self, tensors, opts):
        # FSDP passes a list of tensors
        for t in tensors:
            ops.gin_all_reduce(t, opts.reduceOp, self._pg)
        return dist.Work()  # dummy work handle

    def allgather(self, output_tensors, input_tensors, opts):
        ws = dist.get_world_size(group=self._pg)
        rk = dist.get_rank(group=self._pg)
        for out, inp in zip(output_tensors, input_tensors):
            ops.gin_all_gather_into_tensor(out, inp, ws, rk)
        return dist.Work()

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        for out, inp in zip(output_tensors, input_tensors):
            ops.gin_reduce_scatter_tensor(out, inp, self._pg, opts.reduceOp)
        return dist.Work()
