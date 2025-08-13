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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from ..attention_backend import AttentionMetadata, AttentionType
from ..attention_backend.utils import create_attention


class TriangleAttention(nn.Module):
    """
    A module that implements the triangle attention mechanism with tensor parallelism in torch
    """

    def __init__(self,
                 *,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: Optional[int] = None,
                 layer_idx: int,
                 bias_flags: dict[str, bool] = {
                     "q": False,
                     "k": False,
                     "v": False,
                     "g": False,
                     "z": False,
                     "o": False,
                 },
                 gating: bool = True,
                 dtype: torch.dtype = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 attn_backend: str = "VANILLA"):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.mapping = mapping or Mapping()

        tp_size = self.mapping.tp_size

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=bias_flags["q"] or bias_flags["k"] or bias_flags["v"],
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=bias_flags["o"],
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            skip_create_weights=skip_create_weights,
        )
        self.g_proj = None
        if gating:
            self.g_proj = Linear(
                self.hidden_size,
                tp_size * self.q_size,
                bias=bias_flags["g"],
                dtype=dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=False,
                skip_create_weights=skip_create_weights,
            )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            attention_type=AttentionType.TRIANGLE,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, I, J, F]
            biases: Include two biases:
                - mask_bias: [B, I, 1, 1, J]
                - triangle_bias: [B, H, J, J]
        # TODO: Need to implement DCP here, 1D-mapping, 2D-mapping context
        """
        if self.mapping.tp_size > 1:
            new_biases = []
            new_biases.append(biases[0])
            bias_1_shape = biases[1].shape
            scatter_size = bias_1_shape[1] // self.mapping.tp_size
            start = self.mapping.tp_rank * scatter_size
            end = (self.mapping.tp_rank + 1) * scatter_size
            scatter_bias = biases[1][:, start:end, :]
            new_biases.append(scatter_bias)
            biases = new_biases
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        mha_o = self.attn.forward(q.contiguous(),
                                  k.contiguous(),
                                  v.contiguous(),
                                  biases=biases,
                                  metadata=attn_metadata)
        if self.g_proj is not None:
            g = self.g_proj(hidden_states)
            g = F.sigmoid(g)
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.num_heads, self.head_dim))
            attn_output = mha_o * g
        else:
            attn_output = mha_o
        attn_output = attn_output.view(attn_output.shape[:-2] +
                                       (self.num_heads * self.head_dim, ))
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output


class SelfAttentionPairBias(nn.Module):
    """
    A module that implements the self-attention pair bias mechanism with tensor parallelism in torch.
    This kind of attention is used in the pairformer modules.
    """

    def __init__(self,
                 layer_idx: int,
                 c_s: int,
                 c_z: int,
                 num_heads: int,
                 initial_norm: bool = True,
                 bias_proj: bool = False,
                 dtype: torch.dtype = None,
                 inf: float = 1e6,
                 eps: float = 1e-5,
                 max_attention_pairwise_tp_size: bool = True,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 attn_backend: str = "VANILLA"):
        super().__init__()
        self.layer_idx = layer_idx
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.initial_norm = initial_norm
        self.inf = inf

        self.num_key_value_heads = num_heads
        # This equal to 1 for self-attention
        self.num_key_value_groups = num_heads // self.num_key_value_heads

        mapping = mapping or Mapping()
        if max_attention_pairwise_tp_size:
            mapping = create_max_tp_mapping(mapping, num_heads)
        tp_size = mapping.tp_size
        mapping.tp_rank
        mapping.gpus_per_node

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = self.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.bias_proj = bias_proj

        self.norm_s = None
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s, dtype=dtype, eps=eps)

        self.proj_q = Linear(
            self.c_s,
            tp_size * self.q_size,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
        )
        self.proj_kv = Linear(
            self.c_s,
            2 * tp_size * self.kv_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self.proj_g = Linear(
            self.c_s,
            tp_size * self.q_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
        )
        if self.bias_proj:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z, dtype=dtype, eps=eps),
                Linear(
                    c_z,
                    tp_size * self.num_heads,
                    bias=False,
                    dtype=dtype,
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=False,
                    skip_create_weights=skip_create_weights,
                ),
            )
        self.proj_o = Linear(
            tp_size * self.q_size,
            self.c_s,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            reduce_output=True,
            tensor_parallel_mode=TensorParallelMode.ROW,
            skip_create_weights=skip_create_weights,
        )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            attention_type=AttentionType.PAIRWISE,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        compute_pair_bias: bool = True,
        save_to_cache_key: Optional[str] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        B = s.size(0)
        if self.initial_norm:
            s = self.norm_s(s)
        if not s.is_contiguous():
            s = s.contiguous()
        q = self.proj_q(s)
        kv = self.proj_kv(s)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        mask_bias = (1 - mask[:, None, None].float()) * -self.inf
        pair_bias = z
        if compute_pair_bias and self.bias_proj:
            pair_bias = self.proj_z(z)
            pair_bias = torch.moveaxis(pair_bias, 3,
                                       1)  # [B, N, N, H] -> [B, H, N, N]
        if save_to_cache_key is not None and save_to_cache_key not in attn_metadata.bias_cache:
            attn_metadata.bias_cache[save_to_cache_key] = pair_bias
        biases = [mask_bias, pair_bias]

        mha_o = self.attn.forward(q.contiguous(),
                                  k.contiguous(),
                                  v.contiguous(),
                                  biases=biases,
                                  metadata=attn_metadata)
        o = mha_o.reshape(B, -1, self.num_heads * self.head_dim)

        g = self.proj_g(s).sigmoid()
        o = self.proj_o(g * o, all_reduce_params=all_reduce_params)
        return o


class SelfAttentionPairBiasWithCache(SelfAttentionPairBias):

    def __init__(self, *args, **kwargs):
        kwargs["bias_proj"] = True
        super().__init__(*args, **kwargs)
        self._bias_key = f"{self.__class__.__name__}_{self.layer_idx}"

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                mask: torch.Tensor,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        pair_bias = z
        compute_pair_bias = True
        assert attn_metadata is not None, "attn_metadata is required for self attention pair bias with cache"
        if attn_metadata.bias_cache is not None and self._bias_key in attn_metadata.bias_cache:
            pair_bias = attn_metadata.bias_cache[self._bias_key]
            compute_pair_bias = False
        return super().forward(s=s,
                               z=pair_bias,
                               mask=mask,
                               compute_pair_bias=compute_pair_bias,
                               save_to_cache_key=self._bias_key,
                               attn_metadata=attn_metadata,
                               all_reduce_params=all_reduce_params)


class MSAAttention(nn.Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 c_in: int,
                 num_heads: int,
                 c_z: Optional[int] = None,
                 triangle_attn_backend: str = 'VANILLA',
                 support_batch: bool = True,
                 need_project_z: bool = True,
                 transpose_input: bool = False,
                 bias_flags: dict[str, bool] = {"z": False},
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None,
                 **kwargs):
        super().__init__()
        assert support_batch, "support_batch is required for MSAAttention"
        self.local_layer_idx = local_layer_idx
        self.num_heads = num_heads
        self.c_in = c_in
        self.c_z = c_z
        self.inf = inf
        self.support_batch = support_batch
        self.mapping = mapping or Mapping()
        self.dtype = dtype
        self.transpose_input = transpose_input
        self.triangle_attn_backend = triangle_attn_backend

        self.layer_norm_m = nn.LayerNorm(c_in, dtype=dtype, eps=eps)

        self.proj_z_norm = None
        self.proj_z = None
        if need_project_z:
            self.proj_z_norm = nn.LayerNorm(c_z, dtype=dtype, eps=eps)
            self.proj_z = Linear(
                self.c_z,
                self.num_heads,
                bias=bias_flags["z"],
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                skip_create_weights=skip_create_weights,
                gather_output=True,
            )
        self.mha = TriangleAttention(
            layer_idx=local_layer_idx,
            hidden_size=self.c_in,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            gating=True,
            bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "z": False,
                "o": True,
            },
            dtype=dtype,
            mapping=self.mapping,
            skip_create_weights=skip_create_weights,
            attn_backend=self.triangle_attn_backend,
        )

    def forward(self,
                m: torch.Tensor,
                z: torch.Tensor,
                mask: torch.Tensor,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        """
        Args:
            m: [B, J, I, c_in]
            z: [B, I, I, c_z]
            mask: [B, J, I]
        """
        if self.transpose_input:
            m = torch.permute(m, (0, 2, 1, 3))
            mask = torch.permute(mask, (0, 2, 1))
        mask_bias = ((mask - 1.0) * self.inf)
        mask_bias = mask_bias.unsqueeze(2).unsqueeze(3)

        if self.proj_z_norm and self.proj_z and z is not None:
            z = self.proj_z_norm(z)
            z = self.proj_z(z)
            z = torch.permute(z, (0, 3, 1, 2))  # [B, N, N, H] -> [B, H, N, N]
        else:
            if self.triangle_attn_backend != 'VANILLA':
                # set z to zeros for non-vanilla triangle attention
                z_shape = [m.size(0), self.num_heads, m.size(2), m.size(2)]
                z = torch.zeros(z_shape, dtype=self.dtype).to(m.device)
        biases = [mask_bias, z]
        m = self.layer_norm_m(m)

        output = self.mha(m,
                          biases=biases,
                          attn_metadata=attn_metadata,
                          all_reduce_params=all_reduce_params)
        if self.transpose_input:
            output = torch.permute(output, (0, 2, 1, 3))
        return output
