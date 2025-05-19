# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from tensorrt_bionemo.confs.build_config import BuildModuleConfig
from tensorrt_bionemo.confs.model_config import DimSpec, PretrainedModuleConfig


class PairformerConfig(PretrainedModuleConfig):

    def __init__(self,
                 *,
                 token_s: int = 384,
                 token_z: int = 128,
                 pairwise_head_width: int = 32,
                 pairwise_num_heads: int = 4,
                 num_blocks: int = 48,
                 num_heads: int = 16,
                 max_batch_size: int = 1,
                 max_transition_tp_size: bool = True,
                 max_attention_pairwise_tp_size: bool = True,
                 max_tri_mul_tp_size: bool = True,
                 triangle_attn_node_chunk_size: int = 0,
                 no_update_s: bool = False,
                 no_update_z: bool = False,
                 backend: str = "torch",
                 triangle_attn_backend: str = 'VANILLA',
                 pairwise_attn_backend: str = 'VANILLA',
                 support_batch: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.token_s = token_s
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_batch_size = max_batch_size
        self.max_transition_tp_size = max_transition_tp_size
        self.max_attention_pairwise_tp_size = max_attention_pairwise_tp_size
        self.max_tri_mul_tp_size = max_tri_mul_tp_size
        self.triangle_attn_node_chunk_size = triangle_attn_node_chunk_size
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.backend = backend
        self.triangle_attn_backend = triangle_attn_backend
        self.pairwise_attn_backend = pairwise_attn_backend
        self.disable_custom_all_reduce = max_transition_tp_size or max_attention_pairwise_tp_size
        self.support_batch = support_batch

    def get_input_names(self):
        return list(self.get_input_shapes().keys())

    def get_output_names(self):
        return list(self.get_output_shapes().keys())

    def get_input_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("s", (batch_size, seqlen,
                       DimSpec(size=self.token_s, name="token_s"))),
                ("z", (batch_size, seqlen, seqlen,
                       DimSpec(size=self.token_z, name="token_z"))),
                ("mask", (batch_size, seqlen)),
                ("pair_mask", (batch_size, seqlen, seqlen)),
            ])
        return OrderedDict([
            ("s", (seqlen, DimSpec(size=self.token_s, name="token_s"))),
            ("z", (seqlen, seqlen, DimSpec(size=self.token_z, name="token_z"))),
            ("mask", (seqlen, )),
            ("pair_mask", (seqlen, seqlen)),
        ])

    def get_output_shapes(self):
        seqlen = DimSpec(name="seqlen", dynamic=True)

        if self.support_batch:
            batch_size = DimSpec(name="batch_size", dynamic=True)
            return OrderedDict([
                ("output_s", (batch_size, seqlen,
                              DimSpec(size=self.token_s, name="token_s"))),
                ("output_z", (batch_size, seqlen, seqlen,
                              DimSpec(size=self.token_z, name="token_z"))),
            ])
        return OrderedDict([
            ("output_s", (seqlen, DimSpec(size=self.token_s, name="token_s"))),
            ("output_z", (seqlen, seqlen,
                          DimSpec(size=self.token_z, name="token_z"))),
        ])

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


@dataclass
class PairformerBuildConfig(BuildModuleConfig):
    max_seqlen: int = 128
    min_seqlen: int = 64
    align: int = 32

    @property
    def optimization_profiles(self) -> list[Any]:
        input_shapes = self.module_config.get_input_shapes()

        if self.force_num_profiles == 0:
            return []

        min_seqlen = self.min_seqlen - self.min_seqlen % self.align
        max_seqlen = (self.max_seqlen + self.align -
                      1) // self.align * self.align
        assert (max_seqlen - min_seqlen) % self.force_num_profiles == 0
        step = (max_seqlen - min_seqlen) // self.force_num_profiles

        min_max_seqlens = []

        for i in range(self.force_num_profiles):
            min_max_seqlens.append(
                (min_seqlen + i * step, min_seqlen + (i + 1) * step))

        profiles = []
        for rmin, rmax in min_max_seqlens:
            profile = {}
            for k, v in input_shapes.items():
                min_shape = []
                opt_shape = []
                max_shape = []

                for spec in v:
                    if spec.name == "seqlen":
                        min_shape.append(rmin)
                        opt_shape.append(rmax)
                        max_shape.append(rmax)
                    elif spec.name == "batch_size":
                        min_shape.append(1)
                        opt_shape.append(self.module_config.max_batch_size)
                        max_shape.append(self.module_config.max_batch_size)
                    else:
                        min_shape.append(spec.size)
                        opt_shape.append(spec.size)
                        max_shape.append(spec.size)
                profile[k] = (min_shape, opt_shape, max_shape)
            profiles.append(profile)
        return profiles
