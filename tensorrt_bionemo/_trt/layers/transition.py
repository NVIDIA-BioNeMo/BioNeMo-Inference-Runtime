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

from typing import Optional

import tensorrt as trt
from tensorrt_llm_lite.functional import (Tensor, activation, concat, relu,
                                          silu, split, swiglu)
from tensorrt_llm_lite.layers.linear import Linear
from tensorrt_llm_lite.layers.normalization import LayerNorm
from tensorrt_llm_lite.module import Module, ModuleList

from tensorrt_bionemo._trt.layers.normalization import AdaLN


class Transition(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 dim: int = 128,
                 hidden: int = 512,
                 out_dim: Optional[int] = None,
                 eps: float = 1e-05,
                 dtype: str = None) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.local_layer_idx = local_layer_idx
        self.dim = dim
        self.hidden = hidden
        self.out_dim = out_dim

        self.norm = LayerNorm(normalized_shape=[dim], eps=eps, dtype=dtype)
        self.fused_fc2_fc1 = Linear(self.dim,
                                    2 * self.hidden,
                                    bias=False,
                                    dtype=dtype)
        self.fc3 = Linear(self.hidden, self.out_dim, bias=False, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fused_fc2_fc1(x)
        x = swiglu(x)
        x = self.fc3(x)
        return x


class ConditionedTransitionBlock(Module):
    """Algorithm 25"""

    def __init__(self,
                 dim_single: int,
                 dim_single_cond: int,
                 expansion_factor: int = 2,
                 eps: float = 1e-5,
                 dtype: str = None,
                 using_silu: bool = False):  # using silu instead of swiglu
        super().__init__()

        self.using_silu = using_silu

        self.dim_single = dim_single
        self.dim_single_cond = dim_single_cond
        self.expansion_factor = expansion_factor

        self.adaln = AdaLN(dim_single, dim_single_cond, eps=eps, dtype=dtype)
        self.dim_inner = int(dim_single * expansion_factor)
        # Fused swiglu_gate linear and a_to_b
        if not using_silu:
            # Boltz1, Boltz2 uses swiglu instead of silu
            self.fused_swl_a_to_b = Linear(self.dim_single,
                                           self.dim_inner * 3,
                                           bias=False,
                                           dtype=dtype,
                                           is_qkv=True)
        else:
            # OF3 uses silu instead of swiglu, so we need to use a different column linear
            self.fused_swl_a_to_b = Linear(self.dim_single,
                                           self.dim_inner * 2,
                                           bias=False,
                                           dtype=dtype,
                                           is_qkv=True)
        self.b_to_a = Linear(self.dim_inner,
                             self.dim_single,
                             bias=False,
                             dtype=dtype)
        self.output_projection = Linear(self.dim_single_cond,
                                        self.dim_single,
                                        bias=True,
                                        dtype=dtype)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]

        Returns:
            a: [B, I, d]
        """
        a = self.adaln(a, s)
        z = self.fused_swl_a_to_b(a)
        if not self.using_silu:
            m, n = split(z, [self.dim_inner * 2, self.dim_inner], dim=-1)
            b = swiglu(m) * n  # TODO: Fused swiglu here
        else:
            m, n = split(z, [self.dim_inner, self.dim_inner], dim=-1)
            b = silu(m) * n
        a = self.output_projection(s)
        a = activation(a, trt.ActivationType.SIGMOID) * self.b_to_a(b)
        return a


class PairwiseConditioning(Module):
    """Algorithm 21"""

    def __init__(self,
                 token_z: int,
                 dim_token_rel_pos_feats: int,
                 num_transitions: int = 2,
                 transition_expansion_factor: int = 2,
                 eps: float = 1e-5,
                 dtype: str = None):
        super().__init__()
        self.dtype = dtype
        self.token_z = token_z
        self.dim_token_rel_pos_feats = dim_token_rel_pos_feats
        self.num_transitions = num_transitions

        self.init_proj_norm = LayerNorm(
            normalized_shape=[token_z + dim_token_rel_pos_feats],
            eps=eps,
            dtype=dtype)

        self.init_proj_linear = Linear(token_z + dim_token_rel_pos_feats,
                                       token_z,
                                       bias=False,
                                       dtype=dtype)

        transitions = []
        for i in range(num_transitions):
            transitions.append(
                Transition(local_layer_idx=i,
                           dim=token_z,
                           hidden=token_z * transition_expansion_factor,
                           eps=eps,
                           dtype=self.dtype))
        self.transitions = ModuleList(transitions)

    def forward(self, z_trunk: Tensor, token_rel_pos_feats: Tensor) -> Tensor:
        """
        Args:
            z_trunk: [B, I, I, token_z]
            token_rel_pos_feats: [B, I, I, 3]
        Return:
            [B, I, I, token_z]
        """
        z = concat([z_trunk, token_rel_pos_feats], dim=-1)
        z = self.init_proj_norm(z)
        z = self.init_proj_linear(z)

        for transition in self.transitions:
            z = transition(z) + z
        return z


class PairTransition(Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z: int, n: int, dtype: str, eps: float = 1e-5):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super().__init__()
        self.dtype = dtype
        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm([self.c_z], dtype=dtype, eps=eps)
        self.linear_1 = Linear(self.c_z,
                               self.n * self.c_z,
                               bias=True,
                               dtype=dtype)
        self.linear_2 = Linear(self.n * self.c_z,
                               self.c_z,
                               bias=True,
                               dtype=dtype)

    def forward(self, z: Tensor, mask: Tensor):
        mask = mask.unsqueeze(-1)
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        z = z * mask

        return z


class MSATransition(Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_m: int, n: int, dtype: str, eps: float = 1e-5):
        """
        Args:
            c_m:
                Pair transition channel dimension
            n:
                Factor by which c_m is multiplied to obtain hidden channel
                dimension
        """
        super().__init__()
        self.dtype = dtype
        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm([self.c_m], dtype=dtype, eps=eps)
        self.linear_1 = Linear(self.c_m,
                               self.n * self.c_m,
                               bias=True,
                               dtype=dtype)
        self.linear_2 = Linear(self.n * self.c_m,
                               self.c_m,
                               bias=True,
                               dtype=dtype)

    def forward(self, m: Tensor, mask: Tensor):
        # Similar to PairTransition, but with different names
        mask = mask.unsqueeze(-1)
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = relu(m)
        m = self.linear_2(m)
        m = m * mask

        return m
