# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch
from pydantic import field_serializer, model_validator

from bionemo_ir.utils import torch_dtype_to_str

from .base import BaseConfig


class PairformerConfig(BaseConfig):
    token_s: int | None = None
    token_z: int = None
    pairwise_head_width: int = None
    pairwise_num_heads: int = None
    num_blocks: int = None
    num_heads: int = None
    no_update_s: bool = False
    no_update_z: bool = False
    s_path_dtype: str | torch.dtype | None = None
    post_layer_norm: bool | None = False
    triangle_attn_cueq_fallback_threshold: int = 0
    trimul_high_precision: bool = False
    trimul_mean_normalization: bool = False
    attention_initial_norm: bool | None = True
    version: str = "v1"

    @field_serializer("s_path_dtype")
    @classmethod
    def _serialize_s_path_dtype(cls, v):
        if isinstance(v, torch.dtype):
            return torch_dtype_to_str(v)
        return v

    def set_dtype(self, value: str | torch.dtype) -> None:
        super().set_dtype(value)
        if self.s_path_dtype is None:
            self.s_path_dtype = self.torch_dtype

    @model_validator(mode="after")
    def fill_s_path_dtype(self) -> "PairformerConfig":
        if self.s_path_dtype is None:
            self.s_path_dtype = self.torch_dtype
        return self


class DiffusionTransformerConfig(BaseConfig):
    num_blocks: int = None
    num_heads: int = None
    dim: int = None
    dim_single_cond: int = None
    dim_pairwise: int | None = None
    expansion_factor: int = None
    multiplicity: int | None = 1
    attention_initial_norm: bool | None = None
    post_layer_norm: bool | None = None
    conditioned_transition_using_silu: bool | None = None
    bias_proj: bool | None = None
    # When True, load_weights expects a single "layer_norm_z" weight entry
    # (matching the reference where one LayerNorm is shared across all blocks)
    # and broadcasts it to every block's proj_z.0.  Other modules that use
    # independent per-block LayerNorms should leave this False (default).
    shared_pair_norm: bool = False
    attn_output_gate: bool = True
    attn_gate_bias: bool = False
    transition_expansion_factor: int = 2
    precompute_bias: bool = True


class EvoformerStackConfig(BaseConfig):
    c_m: int = None
    c_z: int = None
    c_s: int = None
    c_hidden_msa_att: int = None
    c_hidden_opm: int = None
    c_hidden_mul: int = None
    c_hidden_pair_att: int = None
    no_heads_msa: int = None
    no_heads_pair: int = None
    transition_n: int = None
    no_blocks: int = None
    no_column_attention: bool = False
    opm_first: bool = False
    n_seq: int = 516
    trimul_high_precision: bool = False
