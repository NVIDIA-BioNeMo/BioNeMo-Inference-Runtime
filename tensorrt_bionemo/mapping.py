# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List


class Mapping(object):

    def __init__(self,
                 world_size: int = 1,
                 rank: int = 0,
                 gpus_per_node: int = 8,
                 dcp_size: int = 1,
                 tp_size: int = 1,
                 pp_size: int = 1):
        """
        This class is similar to the `Mapping` class in `tensorrt_llm.mapping`.
        But it is designed for Bionemo structure prediction models.
        Args:
            world_size (int): total number of GPUs
            rank (int): global rank of the current GPU
            gpus_per_node (int): number of GPUs per node
            dcp_size (int): number of distributed context parallel groups
            tp_size (int): number of tensor parallel groups
            pp_size (int): number of pipeline parallel groups
        """
        # pp_size is always 1 for Bionemo
        if tp_size * pp_size * dcp_size != world_size:
            raise ValueError(
                f"tp_size * pp_size * dcp_size must be equal to world_size,\
                              but got {tp_size} * {pp_size} * {dcp_size} = {tp_size * pp_size * dcp_size} != {world_size}"
            )
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dcp_size = dcp_size
        self.cp_size = dcp_size  # this work-around it to avoid the error for ipc_memory in tensorrt_llm
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.enable_attention_dp = False  # FIXME: remove this after the attention dp is implemented
        self.pp_groups = []
        self.dcp_groups = []
        self.tp_groups = []

        # init pp groups
        for i in range(tp_size * dcp_size):
            ranks = range(i, world_size, tp_size * dcp_size)
            self.pp_groups.append(list(ranks))

        # init dp groups
        for i in range(pp_size):
            for j in range(tp_size):
                ranks = range(i * tp_size * dcp_size + j,
                              (i + 1) * tp_size * dcp_size, tp_size)
                self.dcp_groups.append(list(ranks))

        # init tp groups
        for i in range(pp_size):
            for j in range(dcp_size):
                ranks = range(i * tp_size * dcp_size + j * tp_size,
                              i * tp_size * dcp_size + (j + 1) * tp_size)
                self.tp_groups.append(list(ranks))

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return (self.world_size == other.world_size and self.rank == other.rank
                and self.gpus_per_node == other.gpus_per_node
                and self.tp_size == other.tp_size
                and self.dcp_size == other.dcp_size
                and self.pp_size == other.pp_size)

    def __hash__(self):
        return (hash(self.world_size) ^ hash(self.rank)
                ^ hash(self.gpus_per_node) ^ hash(self.tp_size)
                ^ hash(self.dcp_size) ^ hash(self.pp_size))

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        if not isinstance(rank, int) or rank < 0 or rank >= self.world_size:
            raise ValueError(
                f"Rank should be an integer between 0 and {self.world_size-1}, but got {rank}."
            )
        self._rank = rank

    @property
    def tp_rank(self):
        return self.rank % self.tp_size

    @property
    def dcp_rank(self):
        return self.rank % (self.tp_size * self.dcp_size) // self.tp_size

    @property
    def cp_rank(self):
        return self.dcp_rank  # this work-around it to avoid the error for ipc_memory in tensorrt_llm

    @property
    def pp_rank(self):
        return self.rank // (self.tp_size * self.dcp_size)

    @property
    def tp_group(self):
        return self.tp_groups[self.pp_rank * self.dcp_size + self.dcp_rank]

    @property
    def dcp_group(self):
        return self.dcp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def pp_group(self):
        return self.pp_groups[self.dcp_rank * self.tp_size + self.tp_rank]

    @property
    def node_rank(self):
        return self.rank // self.gpus_per_node

    @property
    def local_rank(self):
        return self.rank % self.gpus_per_node

    def get_node_rank(self, rank: int):
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int):
        return rank % self.gpus_per_node

    def has_dp(self):
        return self.dcp_size > 1

    def has_tp(self):
        return self.tp_size > 1

    def has_pp(self):
        return self.pp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def prev_pp_rank(self):
        p = self.rank - self.tp_size * self.dcp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size * self.dcp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def prev_dcp_rank(self, step: int = 1):
        """ This function is used to get the previous dp rank for ring reduce """
        p = self.rank - self.tp_size * step
        if p < 0:
            p = p % self.world_size
        return p

    def next_dcp_rank(self, step: int = 1):
        """ This function is used to get the next dp rank for ring reduce """
        p = self.rank + self.tp_size * step
        if p >= self.world_size:
            p = p % self.world_size
        return p

    def pp_layers(self, num_layers: int) -> List[int]:
        layers_per_pipeline_stage = num_layers // self.pp_size
        layers_range = range(self.pp_rank * layers_per_pipeline_stage,
                             (self.pp_rank + 1) * layers_per_pipeline_stage)
        return list(layers_range)

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(**mapping)

    def to_dict(self):
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'gpus_per_node': self.gpus_per_node,
            'dcp_size': self.dcp_size,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
        }


def create_max_tp_mapping(mapping: Mapping, dim: int):
    max_tp_size = min(mapping.world_size, dim)
    dcp_size = mapping.world_size // max_tp_size
    while True:
        if dim % max_tp_size == 0:
            return Mapping(world_size=mapping.world_size,
                           rank=mapping.rank,
                           gpus_per_node=mapping.gpus_per_node,
                           dcp_size=dcp_size,
                           tp_size=max_tp_size)
        max_tp_size //= 2
        dcp_size *= 2
