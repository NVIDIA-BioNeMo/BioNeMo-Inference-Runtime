# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Protenix template embedder (AF3 Algorithm 16)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerNoSeqModule
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.utils import str_dtype_to_torch


class ProtenixTemplateEmbedder(nn.Module):
    """Protenix template embedder (AF3 Algorithm 16).

    Shared :class:`PairformerNoSeqModule` for the inner pair stack; per-template
    feature construction / projections stay Protenix-specific. Inner stack runs
    at ``pairformer_dtype`` (bf16); outer projections follow ``dtype``.
    Tri-mul uses stack bf16 (``trimul_high_precision=False``).
    """

    # OSS Algorithm 16 template feature widths (concatenation order matters —
    # it must match linear_no_bias_a weights).
    input_feature1: dict[str, int] = {
        "template_distogram": 39,
        "template_backbone_frame_mask": 1,
        "template_unit_vector": 3,
        "template_pseudo_beta_mask": 1,
    }
    input_feature2: dict[str, int] = {
        "template_restype_i": 32,
        "template_restype_j": 32,
    }
    n_restypes: int = 32  # len(STD_RESIDUES_WITH_GAP)

    # Checkpoint layout of linear_no_bias_a weight columns (split-projection path).
    _DGRAM_END: int = input_feature1["template_distogram"]
    _PSEUDO_BETA_COL: int = _DGRAM_END
    _RESTYPE_I: slice = slice(_PSEUDO_BETA_COL + 1, _PSEUDO_BETA_COL + 1 + n_restypes)
    _RESTYPE_J: slice = slice(_RESTYPE_I.stop, _RESTYPE_I.stop + n_restypes)
    _UNIT_VECTOR: slice = slice(_RESTYPE_J.stop, _RESTYPE_J.stop + input_feature1["template_unit_vector"])
    _BACKBONE_COL: int = _UNIT_VECTOR.stop

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.n_blocks = config.n_blocks
        self.c = config.c
        self.c_z = config.c_z
        self.dtype = config.torch_dtype
        self.pairformer_dtype = str_dtype_to_torch(config.pairformer_dtype)
        a_in = sum(self.input_feature1.values()) + sum(self.input_feature2.values())

        self.layernorm_z = nn.LayerNorm(self.c_z, eps=config.norm_epsilon, dtype=self.dtype)
        self.linear_no_bias_z = Linear(
            self.c_z, self.c, bias=False, dtype=self.dtype, skip_create_weights=config.skip_create_weights
        )
        self.linear_no_bias_a = Linear(
            a_in, self.c, bias=False, dtype=self.dtype, skip_create_weights=config.skip_create_weights
        )
        self.pairformer_stack = PairformerNoSeqModule(
            num_blocks=self.n_blocks,
            token_z=self.c,
            pairwise_head_width=config.pairwise_head_width,
            pairwise_num_heads=config.pairwise_num_heads,
            pair_transition_factor=config.num_intermediate_factor,
            dtype=self.pairformer_dtype,
            eps=config.norm_epsilon,
            inf=config.mask_inf,
            triangle_attn_backend=config.triangle_attention_backend,
            trimul_high_precision=config.trimul_high_precision,
            skip_create_weights=config.skip_create_weights,
        )
        self.layernorm_v = nn.LayerNorm(self.c, eps=config.norm_epsilon, dtype=self.dtype)
        self.linear_no_bias_u = Linear(
            self.c, self.c_z, bias=False, dtype=self.dtype, skip_create_weights=config.skip_create_weights
        )

    def _project_single_template_features(
        self, input_feature_dict: dict[str, Any], template_id: int, masked_by: torch.Tensor
    ) -> torch.Tensor:
        """Project one template without materializing ``[B, N, N, 108]``.

        ``linear_no_bias_a`` layout: ``[dgram, pseudo_beta, restype_i,
        restype_j, unit_vector, backbone]``. Apply weight slices separately.
        """

        def feat(name: str) -> torch.Tensor:
            return input_feature_dict[name][:, template_id]

        linear = self.linear_no_bias_a
        weight = linear.weight
        dtype = weight.dtype
        unit_w = self._UNIT_VECTOR.stop - self._UNIT_VECTOR.start

        # These four feature groups all carry masked_by — project, sum, mask once.
        projected = linear.apply_linear(
            feat("template_distogram").to(dtype),
            weight[:, : self._DGRAM_END],
            None,
        )
        projected.addcmul_(
            feat("template_pseudo_beta_mask").to(dtype).unsqueeze(-1),
            weight[:, self._PSEUDO_BETA_COL],
        )
        projected.reshape(-1, projected.shape[-1]).addmm_(
            feat("template_unit_vector").to(dtype).reshape(-1, unit_w),
            weight[:, self._UNIT_VECTOR].t(),
        )
        projected.addcmul_(
            feat("template_backbone_frame_mask").to(dtype).unsqueeze(-1),
            weight[:, self._BACKBONE_COL],
        )
        projected.mul_(masked_by.to(dtype).unsqueeze(-1))

        # one_hot(restype) @ W == embedding(restype, W.T). restype_i varies on
        # the second pair axis, restype_j on the first.
        aatype = feat("template_aatype").long()
        projected.add_(F.embedding(aatype, weight[:, self._RESTYPE_I].t()).unsqueeze(1))
        projected.add_(F.embedding(aatype, weight[:, self._RESTYPE_J].t()).unsqueeze(2))

        return projected

    def forward(
        self, input_feature_dict: dict[str, Any], z: torch.Tensor, pair_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Template pair update from ``N_templ`` features.

        Args:
            input_feature_dict: ``template_aatype`` ``[B, N_templ, N_token]``,
                ``template_distogram`` ``[B, N_templ, N, N, 39]``,
                ``template_unit_vector`` ``[B, N_templ, N, N, 3]``,
                ``template_pseudo_beta_mask`` /
                ``template_backbone_frame_mask`` ``[B, N_templ, N, N]``,
                plus ``asym_id`` ``[B, N_token]``
            z: ``[B, N_token, N_token, c_z]``

        Returns:
            ``[B, N_token, N_token, c_z]`` (zeros when no templates)
        """
        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            return z.new_zeros(z.shape)

        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[..., :, None] == asym_id[..., None, :]).to(z.dtype)  # [B, N, N]
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])
        masked_by = multichain_mask * pair_mask

        num_templates = input_feature_dict["template_aatype"].shape[1]
        z = self.layernorm_z(z)
        z_proj = self.linear_no_bias_z(z)

        u = z.new_zeros((*z.shape[:-1], self.c))
        for template_id in range(num_templates):
            v = self._project_single_template_features(input_feature_dict, template_id, masked_by)
            v.add_(z_proj)
            v = self.pairformer_stack(z=v.to(self.pairformer_dtype), pair_mask=pair_mask)
            u = u + self.layernorm_v(v.to(self.dtype))
        u = u / (1e-7 + num_templates)
        return self.linear_no_bias_u(F.relu(u))
