import atexit
import enum
import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy

from tensorrt_bionemo.mapping import Mapping

# Because the mapping of bionemo is different from tensorrt_llm
# we need to patch the distributed codes here,
# the original code from tensorrt_llm supports only tensor parallel
# and the data parallel is not supported


class AllGatherMode(str, enum.Enum):
    TP = 'TP'  # tensor parallel
    DP = 'DP'  # data parallel


def allgather(input: torch.Tensor,
              mapping: Mapping,
              gather_dim: int = -1,
              mode: AllGatherMode = AllGatherMode.TP) -> torch.Tensor:
    """ Support both tensor parallel and data parallel """
    if mapping.tp_size == 1 and mode == AllGatherMode.TP:
        return input
    if mapping.dcp_size == 1 and mode == AllGatherMode.DP:
        return input
    if mode == AllGatherMode.TP:
        output = torch.ops.trtllm.allgather(
            input,
            mapping.tp_group,
        )
        split_size = mapping.tp_size
    elif mode == AllGatherMode.DP:
        output = torch.ops.trtllm.allgather(
            input,
            mapping.dcp_group,
        )
        split_size = mapping.dcp_size
    else:
        raise ValueError(f"Invalid allgather mode: {mode}")

    if gather_dim < 0:
        gather_dim += input.ndim
    output = torch.movedim(output, 0, gather_dim)
    input_shape = input.size()
    output = output.reshape(input_shape[:gather_dim] +
                            (split_size * input_shape[gather_dim], ) +
                            input_shape[gather_dim + 1:])
    return output


class AllReduce(nn.Module):

    def __init__(self,
                 mapping: Mapping,
                 strategy: AllReduceStrategy = AllReduceStrategy.NCCL):
        super().__init__()
        """
        AllReduce is a module that performs an all-reduce operation on a tensor.
        Modified from tensorrt_llm, disable workspace.
        """

        self.mapping = mapping
        self.workspace = None
        self.strategy = strategy

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.mapping.tp_size == 1 or (all_reduce_params is not None
                                         and all_reduce_params.enable_allreduce
                                         == False):
            return input

        # Assume using no fusion allreduce here
        if all_reduce_params is None:
            all_reduce_params = AllReduceParams()

        output = torch.ops.trtllm.allreduce(
            input=input,
            residual=all_reduce_params.residual,
            norm_weight=all_reduce_params.norm_weight,
            scale=all_reduce_params.scale,
            bias=all_reduce_params.bias,
            workspace=self.workspace,
            group=self.mapping.tp_group,
            strategy=self.strategy,
            op=all_reduce_params.fusion_op,
            eps=all_reduce_params.eps,
        )

        return output if len(output) > 1 else output[0]


class DPComm:
    # DP communication using torch.distributed with nccl backend
    def __init__(self, global_mapping: Mapping):
        self.mapping = global_mapping
        if not dist.is_initialized():
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="nccl",
                                    init_method=init_method,
                                    world_size=global_mapping.world_size,
                                    rank=global_mapping.rank)
            atexit.register(self._cleanup)

        dist.barrier(device_ids=[torch.cuda.current_device()])

    def _cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def batch_isend_irecv(self,
                          send_tensor,
                          recv_tensor,
                          src=1,
                          dest=1,
                          tag=0) -> torch.Tensor:
        dest_rank = self.mapping.next_dcp_rank(step=dest)
        src_rank = self.mapping.prev_dcp_rank(step=src)
        send_op = dist.P2POp(dist.isend, send_tensor, dest_rank, tag=tag)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank, tag=tag)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()


class DPCommManager:
    _dp_comm = None

    def __init__(self):
        self.tag = 1234

    @classmethod
    def init_dp_comm(cls, mapping):
        """Initialize DPComm once at startup"""
        if cls._dp_comm is None:
            cls._dp_comm = DPComm(mapping)

    @classmethod
    def get_dp_comm(cls):
        """Get the DPComm instance"""
        return cls._dp_comm

    def batch_isend_irecv(self,
                          send_tensor,
                          recv_tensor,
                          src=1,
                          dest=1) -> torch.Tensor:
        return self._dp_comm.batch_isend_irecv(send_tensor, recv_tensor, src,
                                               dest, self.tag)
