# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import math
import weakref
from datetime import timedelta
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from tensorrt_bionemo.logger import logger
from tensorrt_bionemo.mapping import Mapping


class AllReduceParams:
    pass


def init_distributed_environment(world_size: int = -1,
                                 rank: int = -1,
                                 init_method: str = "env://",
                                 backend: str = "nccl",
                                 timeout: timedelta | None = None,
                                 device_id: Optional[int] = None) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend=backend,
                                init_method=init_method,
                                world_size=world_size,
                                rank=rank,
                                device_id=device_id)


_groups: dict[str, Callable[[], Optional["GroupCoordinator"]]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


_group_name_counter: dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


class GroupCoordinator(object):
    """ This implementation a very basic group coordinator using naive torch.distributed.
    In the vLLM or Megatron-LM, the class extends for custom collective operations. """
    rank: int
    ranks: list[int]
    world_size: int
    local_rank: int
    rank_in_group: int
    device_group: dist.ProcessGroup

    # cpu_group: dist.ProcessGroup

    def __init__(self,
                 group_ranks: list[int],
                 local_rank: int,
                 group_name: str | None = None,
                 torch_distributed_backend: str = "nccl"):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        device_group = torch.distributed.new_group(
            group_ranks, backend=torch_distributed_backend)

        self.ranks = group_ranks
        self.world_size = len(group_ranks)
        self.rank_in_group = group_ranks.index(self.rank)
        self.device_group = device_group
        # self.cpu_group = torch.distributed.new_group(group_ranks, backend="gloo")
        self.device = torch.device(f"cuda:{local_rank}")

    def all_reduce(self,
                   input_: torch.Tensor,
                   op=dist.ReduceOp.SUM,
                   params: AllReduceParams = None) -> torch.Tensor:
        """
        All-reduce operation.
        Args:
            input_: The input tensor to all-reduce.
            op: The operation to perform.
            params: The parameters for the all-reduce operation. Defaults to None. (Reversed for future use)
        Returns:
            The all-reduced tensor.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        out = input_.clone()
        dist.all_reduce(out, op=op, group=self.device_group)
        return out

    def _get_output_info(self, input_: torch.Tensor, dim: int) -> List[int]:
        dim = dim % input_.ndim
        output_shape = [
            val if idx != dim else -1 for idx, val in enumerate(input_.shape)
        ]
        numel_base = -math.prod(output_shape)
        return {'output_shape': output_shape, 'numel_base': numel_base}

    def all_gather(self, input_: torch.Tensor, dim=-1) -> torch.Tensor:
        output_info = self._get_output_info(input_, dim)
        input_ = input_.contiguous().view(-1, output_info['numel_base'])
        gathered = torch.empty(self.world_size,
                               *input_.shape,
                               device=input_.device,
                               dtype=input_.dtype)
        dist.all_gather_into_tensor(gathered,
                                    input_,
                                    group=self.device_group,
                                    async_op=False)
        if dim == 0:
            x = gathered.view(output_info['output_shape'])
        else:
            x_list = gathered.chunk(self.world_size)
            x = torch.cat(
                [x.reshape(output_info['output_shape']) for x in x_list],
                dim=dim)

        return x

    def __del__(self):
        logger.debug(
            f"(Deleting group coordinator {self=}) rank: {self.rank}, unique_name: {self.unique_name}"
        )

    @property
    def next_rank(self) -> int:
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self) -> int:
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    def batch_isend_irecv(self,
                          send_tensor: torch.Tensor,
                          recv_tensor: torch.Tensor,
                          tag=123) -> None:
        dest_rank = self.next_rank
        src_rank = self.prev_rank
        send_op = dist.P2POp(dist.isend, send_tensor, dest_rank, tag=tag)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank, tag=tag)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        TODO: Investigate what the reason behind this is for self.cpu_group.
        """
        # torch.distributed.barrier(group=self.cpu_group)
        torch.distributed.barrier(group=self.device_group)


def get_group_coordinator(group_name: str) -> Optional["GroupCoordinator"]:
    return _groups.get(group_name, None)


def get_default_tp_group_coordinator() -> Optional["GroupCoordinator"]:
    return get_group_coordinator("tp:0")


def get_default_dcp_group_coordinator() -> Optional["GroupCoordinator"]:
    return get_group_coordinator("dcp:0")


_TP: GroupCoordinator | None = None
# TODO: This need to be 2-D DCP for CP.
_DCP: GroupCoordinator | None = None


def register_tp_group_coordinator(mapping: Mapping,
                                  group_name: str = "tp"
                                  ) -> "GroupCoordinator":
    global _TP
    if _TP is None:
        _TP = GroupCoordinator(mapping.tp_group, mapping.local_rank,
                               group_name)
    return get_group_coordinator(_TP.unique_name)


def register_dcp_group_coordinator(mapping: Mapping,
                                   group_name: str = "dcp"
                                   ) -> "GroupCoordinator":
    global _DCP
    if _DCP is None:
        _DCP = GroupCoordinator(mapping.dcp_group, mapping.local_rank,
                                group_name)
    return get_group_coordinator(_DCP.unique_name)
