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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerNoSeqModule
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping


class TemplateV2Module(nn.Module):
    """Boltz-2 v2 template module.

    Reference (upstream eager implementation):
    ``the upstream Boltz implementation``.

    The module embeds template features into a ``template_dim``-wide pair
    representation, runs a small ``PairformerNoSeqModule`` over the per-template
    pair representation, aggregates across templates, and projects back to
    ``token_z`` to produce an additive update for the trunk pair embedding.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.mapping = config.mapping or Mapping()
        self.dtype = config.torch_dtype

        self.token_z = config.token_z
        self.template_dim = config.template_dim
        self.template_blocks = config.template_blocks
        self.min_dist = config.min_dist
        self.max_dist = config.max_dist
        self.num_bins = config.num_bins
        self.num_tokens = config.num_tokens

        skip_create_weights = config.skip_create_weights

        self.z_norm = nn.LayerNorm(self.token_z,
                                   eps=config.norm_epsilon,
                                   dtype=self.dtype)
        # ``v_norm`` lives on the inner ``template_dim`` representation, so
        # it follows the inner pairformer's dtype.
        self.v_norm = nn.LayerNorm(self.template_dim,
                                   eps=config.norm_epsilon,
                                   dtype=config.pairformer.torch_dtype)

        self.z_proj = Linear(self.token_z,
                             self.template_dim,
                             bias=False,
                             dtype=self.dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=skip_create_weights)
        a_in = self.num_tokens * 2 + self.num_bins + 5
        self.a_proj = Linear(a_in,
                             self.template_dim,
                             bias=False,
                             dtype=self.dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=skip_create_weights)
        self.u_proj = Linear(self.template_dim,
                             self.token_z,
                             bias=False,
                             dtype=self.dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=skip_create_weights)

        self.pairformer = PairformerNoSeqModule(
            num_blocks=self.template_blocks,
            token_z=self.template_dim,
            pairwise_head_width=config.pairwise_head_width,
            pairwise_num_heads=config.pairwise_num_heads,
            dtype=config.pairformer.torch_dtype,
            eps=config.norm_epsilon,
            inf=config.mask_inf,
            mapping=self.mapping,
            triangle_attn_backend=config.pairformer.triangle_attention_backend,
            skip_create_weights=skip_create_weights,
            attention_initial_norm=config.pairformer.attention_initial_norm,
            trimul_high_precision=config.pairformer.trimul_high_precision,
            trimul_mean_normalization=config.pairformer.
            trimul_mean_normalization,
        )

        self.register_buffer(
            "boundaries",
            torch.linspace(self.min_dist,
                           self.max_dist,
                           self.num_bins - 1,
                           dtype=torch.float32),
            persistent=False,
        )

    def load_weights(self, weights: dict):
        loaded = recursive_calling_load_weights(self, weights)
        missing = set(weights.keys()) - loaded
        if missing:
            raise ValueError(
                f"The following weights are not loaded: {missing}")

    def forward(
        self,
        z: torch.Tensor,
        feats: dict[str, torch.Tensor],
        pair_mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        """Compute the template update for the pair representation.

        Args:
            z: Pair embedding of shape ``(B, N, N, token_z)``.
            feats: Feature dictionary that must contain ``template_restype``,
                ``template_frame_rot``, ``template_frame_t``,
                ``template_mask_frame``, ``template_cb``, ``template_ca``,
                ``template_mask_cb``, ``visibility_ids``, and ``template_mask``.
            pair_mask: Pair mask of shape ``(B, N, N)``.
            attn_metadata: Optional attention metadata forwarded to the
                inner pairformer's triangle attention.
            all_reduce_params: Optional all-reduce parameters for TP.

        Returns:
            Pair update tensor of shape ``(B, N, N, token_z)``.
        """
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        visibility_ids = feats["visibility_ids"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1).clamp(min=1)

        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]
        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        B, T = res_type.shape[:2]  # noqa: N806
        tmlp_pair_mask = (
            visibility_ids[:, :, :, None] == visibility_ids[:, :,
                                                            None, :]).float()

        # Match the upstream module's explicit fp32 numerics for the template
        # geometric features. The bf16 cast happens just before ``a_proj``.
        with torch.autocast(device_type="cuda", enabled=False):
            cb_coords_f = cb_coords.float()
            cb_dists = torch.cdist(cb_coords_f, cb_coords_f)
            boundaries = self.boundaries.to(cb_dists.dtype)
            # ``torch.bucketize`` is the direct expression of
            # ``(cb_dists[..., None] > boundaries).sum(dim=-1).long()`` and
            # avoids materializing the ``(B, T, N, N, num_bins-1)`` boolean
            # comparison tensor + reduction. Returns int64 directly.
            distogram = torch.bucketize(cb_dists, boundaries)
            distogram = F.one_hot(distogram, num_classes=self.num_bins).float()

            frame_rot_f = frame_rot.float().unsqueeze(2).transpose(-1, -2)
            frame_t_f = frame_t.float().unsqueeze(2).unsqueeze(-1)
            ca_coords_f = ca_coords.float().unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot_f, ca_coords_f - frame_t_f)
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm,
                                      torch.zeros_like(vector)).squeeze(-1)

            res_type_f = res_type.float()
            res_i = res_type_f[:, :, :, None].expand(-1, -1, -1,
                                                     res_type.size(2), -1)
            res_j = res_type_f[:, :, None, :].expand(-1, -1, res_type.size(2),
                                                     -1, -1)
            # Single fused concat in feature-dim order:
            #   distogram | b_cb_mask | unit_vector | b_frame_mask | res_i | res_j
            # then mask the geometric portion with ``tmlp_pair_mask`` (the
            # res_i/res_j parts of the upstream code are unmasked, so we
            # mask only the leading geometric slice).
            a_tij_geo = torch.cat([
                distogram,
                b_cb_mask.float(),
                unit_vector,
                b_frame_mask.float(),
            ],
                                  dim=-1)
            a_tij_geo = a_tij_geo * tmlp_pair_mask.unsqueeze(-1)
            a_tij = torch.cat([a_tij_geo, res_i, res_j], dim=-1)

        a_tij = self.a_proj(a_tij.to(self.dtype))

        N = z.shape[1]  # noqa: N806
        pair_mask_t = pair_mask[:, None].expand(-1, T, -1,
                                                -1).reshape(B * T, N, N)

        v = self.z_proj(self.z_norm(z[:, None].to(self.dtype))) + a_tij
        v = v.view(B * T, N, N, self.template_dim)
        # Inner pairformer may run at a different dtype than the outer module.
        inner_dtype = self.pairformer.layers[0].dtype
        v = v.to(inner_dtype)
        v = v + self.pairformer(
            v,
            pair_mask_t.to(inner_dtype),
            attn_metadatas={"triangle_attn": attn_metadata},
            all_reduce_params=all_reduce_params)
        v = self.v_norm(v)
        v = v.view(B, T, N, N, self.template_dim)

        template_mask = template_mask[:, :, None, None, None].to(v.dtype)
        num_templates = num_templates[:, None, None, None].to(v.dtype)
        u = (v * template_mask).sum(dim=1) / num_templates

        u = u.to(self.dtype)
        u = self.u_proj(F.relu(u))
        return u
