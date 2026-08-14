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
import torch.nn as nn

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import PrecomputedPairMasks, precompute_pair_masks
from bionemo_ir._torch.layers.attention import MSAAttention
from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.outer_product_mean import OuterProductMean
from bionemo_ir._torch.layers.transition import MSATransition, PairTransition
from bionemo_ir._torch.layers.triangle_nodes import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
)
from bionemo_ir._torch.utils import recursive_calling_load_weights
from bionemo_ir.configs import BaseConfig
from bionemo_ir.runtime.buffers import PreallocatedBuffers


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        local_layer_idx: int,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        no_column_attention: bool = False,
        opm_first: bool = False,
        triangle_attn_backend: str = "VANILLA",
        support_batch: bool = True,
        dtype: torch.dtype | None = None,
        trimul_high_precision: bool = False,
        pair_mask_left_aligned: bool = True,
        eps: float = 1e-5,
        inf: float = 1e9,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden_msa_att = c_hidden_msa_att
        self.c_hidden_opm = c_hidden_opm
        self.c_hidden_mul = c_hidden_mul
        self.c_hidden_pair_att = c_hidden_pair_att
        self.no_heads_msa = no_heads_msa
        self.no_heads_pair = no_heads_pair
        self.transition_n = transition_n
        self.no_column_attention = no_column_attention
        self.opm_first = opm_first
        self.triangle_attn_backend = triangle_attn_backend
        self.pair_mask_left_aligned = pair_mask_left_aligned
        self.support_batch = support_batch
        self.dtype = dtype
        self.eps = eps
        self.inf = inf
        outer_product_mean_bias = kwargs.get("outer_product_mean_bias", None)
        tri_mul_out_bias = kwargs.get("tri_mul_out_bias", None)
        tri_mul_in_bias = kwargs.get("tri_mul_in_bias", None)
        tri_attn_start_bias = kwargs.get("tri_attn_start_bias", None)
        tri_attn_end_bias = kwargs.get("tri_attn_end_bias", None)

        self.msa_att_row = MSAAttention(
            local_layer_idx=local_layer_idx,
            c_in=c_m,
            num_heads=no_heads_msa,
            c_z=c_z,
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
            need_project_z=True,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.msa_transition = MSATransition(
            c_m=c_m, n=transition_n, eps=eps, dtype=dtype, skip_create_weights=skip_create_weights
        )

        self.outer_product_mean = OuterProductMean(
            c_in=c_m,
            c_hidden=c_hidden_opm,
            c_out=c_z,
            eps=eps,
            mask_eps=1e-3,
            norm_mask_by_eps=True,
            norm_before_output=False,
            cast_to_float_before_einsum=False,
            bias_flags={"proj_a": True, "proj_b": True, "proj_o": True}
            if outer_product_mean_bias is None
            else outer_product_mean_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.tri_mul_out = TriangleMultiplicationNode(
            layer_idx=local_layer_idx,
            dim=c_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            bias_flags={"p_in": True, "g_in": True, "p_out": True, "g_out": True}
            if tri_mul_out_bias is None
            else tri_mul_out_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            high_precision=trimul_high_precision,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )

        self.tri_mul_in = TriangleMultiplicationNode(
            layer_idx=local_layer_idx,
            dim=c_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            bias_flags={"p_in": True, "g_in": True, "p_out": True, "g_out": True}
            if tri_mul_in_bias is None
            else tri_mul_in_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            high_precision=trimul_high_precision,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )

        self.tri_attn_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
            layer_idx=local_layer_idx,
            mha_bias_flags={"q": False, "k": False, "v": False, "g": True, "z": False, "o": True}
            if tri_attn_start_bias is None
            else tri_attn_start_bias,
            attn_backend=triangle_attn_backend,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )
        self.tri_attn_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
            layer_idx=local_layer_idx,
            mha_bias_flags={"q": False, "k": False, "v": False, "g": True, "z": False, "o": True}
            if tri_attn_end_bias is None
            else tri_attn_end_bias,
            attn_backend=triangle_attn_backend,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )

        self.pair_transition = PairTransition(c_z=c_z, n=transition_n, dtype=dtype, eps=eps)
        if not self.no_column_attention:
            self.msa_att_col = MSAAttention(
                local_layer_idx=local_layer_idx,
                c_in=c_m,
                num_heads=no_heads_msa,
                c_z=None,
                triangle_attn_backend=triangle_attn_backend,
                support_batch=support_batch,
                need_project_z=False,
                transpose_input=True,
                eps=eps,
                inf=inf,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )

    def _compute_opm(
        self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opm = self.outer_product_mean(m, mask=msa_mask)
        z = z + opm
        return m, z

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            precomputed_masks:
                Optional precomputed mask biases (avoids redundant computation
                when called inside a layer loop).
            buffers: Shared pre-allocated buffer dict for CuTeDSL kernels.
        """
        if self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)
        m = m + self.msa_att_row(m, z, mask=msa_mask, attn_metadata=attn_metadata)
        if not self.no_column_attention:
            m = m + self.msa_att_col(m, z=None, mask=msa_mask, attn_metadata=attn_metadata)
        m = m + self.msa_transition(m, mask=msa_mask)

        if not self.opm_first:
            m, z = self._compute_opm(m, z, msa_mask)
        # Reuse CuTeDSL's int32 row lengths only for left-aligned masks.
        # Otherwise, let the wrapper derive masking from ``pair_mask``.
        tri_out_actual_seqlen = tri_in_actual_seqlen = None
        if (
            self.pair_mask_left_aligned
            and precomputed_masks is not None
            and precomputed_masks.mask_bias.dtype == torch.int32
        ):
            tri_out_actual_seqlen = precomputed_masks.mask_bias
            tri_in_actual_seqlen = precomputed_masks.mask_bias_transposed
        z = z + self.tri_mul_out(z, mask=pair_mask, actual_seqlen=tri_out_actual_seqlen)
        z = z + self.tri_mul_in(z, mask=pair_mask, actual_seqlen=tri_in_actual_seqlen)

        # Same left-aligned gate as tri_mul: CuTeDSL int32 row lengths are
        # invalid for bipartite / interior-zero masks.
        if self.pair_mask_left_aligned and precomputed_masks is not None:
            mb_start = precomputed_masks.mask_bias
            mb_end = precomputed_masks.mask_bias_transposed
        else:
            mb_start = mb_end = None

        z = z + self.tri_attn_start(z, mask=pair_mask, mask_bias=mb_start, attn_metadata=attn_metadata, buffers=buffers)
        z = z + self.tri_attn_end(z, mask=pair_mask, mask_bias=mb_end, attn_metadata=attn_metadata, buffers=buffers)

        z = z + self.pair_transition(z, mask=pair_mask)

        return m, z


class EvoformerStack(nn.Module):
    def __init__(self, config: BaseConfig):
        """
        Args:
            config: bionemo_ir.configs.modules.EvoformerStackConfig
                The configuration of the evoformer stack module.
        """
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        self.num_blocks = config.no_blocks
        for i in range(self.num_blocks):
            self.blocks.append(
                EvoformerBlock(
                    local_layer_idx=i,
                    c_m=config.c_m,
                    c_z=config.c_z,
                    c_hidden_msa_att=config.c_hidden_msa_att,
                    c_hidden_opm=config.c_hidden_opm,
                    c_hidden_mul=config.c_hidden_mul,
                    c_hidden_pair_att=config.c_hidden_pair_att,
                    no_heads_msa=config.no_heads_msa,
                    no_heads_pair=config.no_heads_pair,
                    transition_n=config.transition_n,
                    no_column_attention=config.no_column_attention,
                    opm_first=config.opm_first,
                    triangle_attn_backend=config.triangle_attention_backend,
                    support_batch=config.support_batch,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    skip_create_weights=config.skip_create_weights,
                    trimul_high_precision=config.trimul_high_precision,
                )
            )
        self.linear = Linear(
            config.c_m, config.c_s, bias=True, dtype=config.torch_dtype, skip_create_weights=config.skip_create_weights
        )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = self.blocks[0].triangle_attn_backend
        precomputed = precompute_pair_masks(
            backend,
            pair_mask,
            inf=self.blocks[0].inf,
            dtype=self.blocks[0].dtype,
        )
        buffers: PreallocatedBuffers | None = {} if backend == "CuTeDSL" else None
        for block in self.blocks:
            m, z = block(m, z, msa_mask, pair_mask, attn_metadata, precomputed_masks=precomputed, buffers=buffers)
        s = self.linear(m[..., 0, :, :])

        return m, z, s
