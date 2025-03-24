""" Copy from tensorrt_llm._torch.distributed.py with some modifications"""
import enum
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
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
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    gpus_per_node: int = 8
    tensor_parallel_mode: Optional[TensorParallelMode] = None
    gather_output: bool = False


def allgather(input: torch.Tensor,
              parallel_config: ParallelConfig,
              gather_dim: int = -1,
              mode: AllGatherMode = AllGatherMode.TP) -> torch.Tensor:
    """ Support both tensor parallel and data parallel """
    if parallel_config.tensor_parallel_size == 1 and parallel_config.data_parallel_size == 1:
        return input

    tp_size = parallel_config.tensor_parallel_size
    dp_size = parallel_config.data_parallel_size
    tp_rank = parallel_config.tensor_parallel_rank
    dp_rank = parallel_config.data_parallel_rank
    mapping = Mapping(
        world_size=tp_size * dp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        rank=dp_rank * tp_size + tp_rank,
        gpus_per_node=parallel_config.gpus_per_node,
    )

    if mode == AllGatherMode.TP:
        output = torch.ops.trtllm.allgather(
            input,
            mapping.tp_group,
        )
        split_size = parallel_config.tensor_parallel_size
    elif mode == AllGatherMode.DP:
        output = torch.ops.trtllm.allgather(
            input,
            mapping.dp_group,
        )
        split_size = parallel_config.data_parallel_size
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
    if parallel_config.tensor_parallel_size == 1 or (
            all_reduce_params is not None
            and all_reduce_params.enable_allreduce == False):
        return input

    tp_size = parallel_config.tensor_parallel_size
    tp_rank = parallel_config.tensor_parallel_rank
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    mapping = Mapping(
        world_size=tp_size * dp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        rank=dp_rank * tp_size + tp_rank,
        gpus_per_node=parallel_config.gpus_per_node,
    )

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
                 strategy: AllReduceStrategy = AllReduceStrategy.NCCL):
        super().__init__()

        self.parallel_config = parallel_config
        self.tp_size = self.parallel_config.tensor_parallel_size
        self.tp_rank = self.parallel_config.tensor_parallel_rank
        self.dp_size = self.parallel_config.data_parallel_size
        self.dp_rank = self.parallel_config.data_parallel_rank
        self.gpus_per_node = self.parallel_config.gpus_per_node

        self.workspace = None
        self.strategy = strategy
        if self.tp_size > 1:
            mapping = Mapping(
                world_size=self.tp_size * self.dp_size,
                tp_size=self.tp_size,
                dp_size=self.dp_size,
                rank=self.dp_rank * self.tp_size + self.tp_rank,
                gpus_per_node=self.gpus_per_node,
            )
            if self.strategy != AllReduceStrategy.UB:
                self.workspace = get_allreduce_workspace(mapping)

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
