""" Copy from tensorrt_llm._torch.distributed.py with some modifications"""
import atexit
import enum
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from tensorrt_llm._torch.distributed import (TensorParallelMode,
                                             get_allreduce_workspace)
from tensorrt_llm.functional import (AllReduceConfig, AllReduceParams,
                                     AllReduceStrategy)
from torch import nn

from tensorrt_bionemo.mapping import Mapping

# Because the mapping of bionemo is different from tensorrt_llm
# we need to patch the distributed codes here,
# the original code from tensorrt_llm supports only tensor parallel
# and the data parallel is not supported


class AllGatherMode(str, enum.Enum):
    TP = 'TP'  # tensor parallel
    DP = 'DP'  # data parallel


@dataclass(kw_only=True)
class ParallelConfig:
    mapping: Mapping
    tensor_parallel_mode: Optional[TensorParallelMode] = None
    gather_output: bool = False


def create_parallel_config(
        mapping: Mapping,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False):
    return ParallelConfig(mapping=deepcopy(mapping),
                          tensor_parallel_mode=tensor_parallel_mode,
                          gather_output=gather_output)


def allgather(input: torch.Tensor,
              parallel_config: ParallelConfig,
              gather_dim: int = -1,
              mode: AllGatherMode = AllGatherMode.TP) -> torch.Tensor:
    """ Support both tensor parallel and data parallel """
    mapping = parallel_config.mapping
    if mapping.tp_size == 1 and mapping.dcp_size == 1:
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


def allreduce(
    input: torch.Tensor,
    workspace: Optional[torch.LongTensor],
    parallel_config: ParallelConfig,
    strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    config: AllReduceConfig = AllReduceConfig(0),
    all_reduce_params: Optional[AllReduceParams] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    mapping = parallel_config.mapping
    if mapping.tp_size == 1 or (all_reduce_params is not None and
                                all_reduce_params.enable_allreduce == False):
        return input

    if all_reduce_params is None:
        all_reduce_params = AllReduceParams()
    reduce_fusion_inputs = []

    final_output, _ = torch.ops.trtllm.allreduce(
        input,
        workspace,
        reduce_fusion_inputs,
        mapping.tp_group,
        int(strategy),
        int(config),
        int(all_reduce_params.fusion_op),
        float(all_reduce_params.eps),
        all_reduce_params.has_affine(),
        all_reduce_params.has_bias(),
        all_reduce_params.has_scale(),
    )

    return final_output


class AllReduce(nn.Module):

    def __init__(self,
                 parallel_config: ParallelConfig,
                 strategy: AllReduceStrategy = AllReduceStrategy.AUTO):
        super().__init__()

        self.parallel_config = parallel_config
        self.mapping = self.parallel_config.mapping
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.gpus_per_node = self.mapping.gpus_per_node

        self.workspace = None
        self.strategy = strategy
        if self.tp_size > 1:
            if self.strategy != AllReduceStrategy.UB:
                self.workspace = get_allreduce_workspace(self.mapping)

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        output = allreduce(input,
                           self.workspace,
                           self.parallel_config,
                           all_reduce_params=all_reduce_params,
                           strategy=self.strategy)
        return output


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

        # Force NCCL initialization and rank population via PyTorch distributed barrier.
        # This is necessary for NOW if using dp + tp because our custom nccl allreduce
        # op for tp groups can interfere with PyTorch's NCCL initialization when PyTorch
        # distributed performs the first comm. op and kick off nccl init. The barrier here
        # ensures proper NCCL setup and GPU-procs binding at beginning.
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
