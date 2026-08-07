# Copyright 2021 DeepMind Technologies Limited
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

from functools import partial

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_bionemo._torch.tensor_utils import dict_multimap, dist_one_hot, tensor_tree_map
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig

from .template import TemplatePairStack, TemplatePointwiseAttention
from .utils import all_atom_multimer, geometry
from .utils.feats import build_template_angle_feat, build_template_pair_feat, dgram_from_positions, pseudo_beta_fn


def relpos(ri: torch.Tensor, boundaries: torch.Tensor, linear_relpos: Linear) -> torch.Tensor:
    """
    Computes relative positional encodings
    Args:
        ri:
            "residue_index" features of shape [*, N]
        boundaries:
            Boundaries of shape [no_bins]
        linear_relpos:
            Linear layer for relative positional encodings
    Returns:
        Relative positional encodings of shape [*, N, N, c_z]
    """
    d = ri[..., None] - ri[..., None, :]
    d = dist_one_hot(d, boundaries).to(ri.dtype)
    return linear_relpos(d)


class InputEmbedder(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.c_z = config.c_z
        self.fused_linear_tf_z = Linear(
            config.tf_dim,
            config.c_z * 2,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self.linear_tf_m = Linear(
            config.tf_dim,
            config.c_m,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

        self.linear_msa_m = Linear(
            config.msa_dim,
            config.c_m,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

        # RPE stuff
        self.relpos_k = config.relpos_k
        self.no_bins = 2 * self.relpos_k + 1
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1)
        self.register_buffer("boundaries", boundaries)
        self.linear_relpos = Linear(
            self.no_bins,
            config.c_z,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def forward(
        self, target_feat: torch.Tensor, residue_index: torch.Tensor, msa_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_feat:
                Features of shape [*, N_res, tf_dim]
            residue_index:
                Features of shape [*, N_res]
            msa_feat:
                Features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb = self.fused_linear_tf_z(target_feat)
        tf_emb_i, tf_emb_j = tf_emb.split([self.c_z, self.c_z], dim=-1)

        # [*, N_res, N_res, c_z]
        pair_emb = relpos(residue_index.to(tf_emb_i), self.boundaries, self.linear_relpos)
        pair_emb = pair_emb + tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]

        # [*, N_clust, N_res, c_m]
        n_clust = msa_feat.shape[-3]
        tf_m = (
            self.linear_tf_m(target_feat).unsqueeze(-3).expand((-1,) * len(target_feat.shape[:-2]) + (n_clust, -1, -1))
        )
        msa_emb = self.linear_msa_m(msa_feat) + tf_m
        return msa_emb, pair_emb


class InputEmbedderMultimer(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.c_z = config.c_z
        self.fused_linear_tf_z = Linear(
            config.tf_dim,
            config.c_z * 2,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self.linear_tf_m = Linear(
            config.tf_dim,
            config.c_m,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

        self.linear_msa_m = Linear(
            config.msa_dim,
            config.c_m,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

        # RPE stuff
        self.max_relative_idx = config.max_relative_idx
        self.use_chain_relative = config.use_chain_relative
        self.max_relative_chain = config.max_relative_chain
        if self.use_chain_relative:
            self.no_bins = 2 * self.max_relative_idx + 2 + 1 + 2 * self.max_relative_chain + 2
            rel_pos_boundaries = torch.arange(start=0, end=2 * self.max_relative_idx + 2)
            self.register_buffer("rel_pos_boundaries", rel_pos_boundaries)
            rel_chain_boundaries = torch.arange(start=0, end=2 * self.max_relative_chain + 2)
            self.register_buffer("rel_chain_boundaries", rel_chain_boundaries)
        else:
            self.no_bins = 2 * self.max_relative_idx + 1
            rel_pos_boundaries = torch.arange(start=0, end=2 * self.max_relative_idx + 1)
            self.register_buffer("rel_pos_boundaries", rel_pos_boundaries)

        self.linear_relpos = Linear(
            self.no_bins,
            config.c_z,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def relpos(
        self, residue_index: torch.Tensor, asym_id: torch.Tensor, entity_id: torch.Tensor, sym_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes relative positional encodings
        Args:
            residue_index:
                Features of shape [*, N_res]
            asym_id:
                Features of shape [*, N_res]
            entity_id:
                Features of shape [*, N_res]
            sym_id:
                Features of shape [*, N_res]
        Returns:
            rel_pos:
            Relative positional encodings of shape [*, N_res, N_res, c_z]
        """
        pos = residue_index
        asym_id_same = asym_id[..., None] == asym_id[..., None, :]
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(offset + self.max_relative_idx, 0, 2 * self.max_relative_idx)

        rel_feats = []

        if self.use_chain_relative:
            final_offset = torch.where(
                asym_id_same, clipped_offset, (2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset)
            )
            rel_pos = dist_one_hot(final_offset, self.rel_pos_boundaries)

            rel_feats.append(rel_pos)

            entity_id_same = entity_id[..., None] == entity_id[..., None, :]
            rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))

            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

            clipped_rel_chain = torch.clamp(
                rel_sym_id + self.max_relative_chain,
                0,
                2 * self.max_relative_chain,
            )

            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * self.max_relative_chain + 1) * torch.ones_like(clipped_rel_chain),
            )

            rel_chain = dist_one_hot(
                final_rel_chain,
                self.rel_chain_boundaries,
            )
            rel_feats.append(rel_chain)
        else:
            rel_pos = dist_one_hot(
                clipped_offset,
                self.rel_pos_boundaries,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(self.config.torch_dtype)
        return self.linear_relpos(rel_feat)

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        msa_feat: torch.Tensor,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_feat:
                Features of shape [*, N_res, tf_dim]
            residue_index:
                Features of shape [*, N_res]
            msa_feat:
                Features of shape [*, N_clust, N_res, msa_dim]
            asym_id:
                Features of shape [*, N_res]
            entity_id:
                Features of shape [*, N_res]
            sym_id:
                Features of shape [*, N_res]
        Returns:
            msa_emb:
                [*, N_clust, N_res, c_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, c_z] pair embedding
        """

        # [*, N_res, c_z]
        tf_emb = self.fused_linear_tf_z(target_feat)
        tf_emb_i, tf_emb_j = tf_emb.split([self.c_z, self.c_z], dim=-1)

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(residue_index, asym_id, entity_id, sym_id)

        # [*, N_clust, N_res, c_m]
        n_clust = msa_feat.shape[-3]
        tf_m = (
            self.linear_tf_m(target_feat).unsqueeze(-3).expand((-1,) * len(target_feat.shape[:-2]) + (n_clust, -1, -1))
        )
        msa_emb = self.linear_msa_m(msa_feat) + tf_m

        return msa_emb, pair_emb


class RecyclingEmbedder(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.c_m = config.c_m
        self.c_z = config.c_z
        self.min_bin = config.min_bin
        self.max_bin = config.max_bin
        self.no_bins = config.no_bins
        self.inf = config.mask_inf

        self.linear = Linear(
            self.no_bins, self.c_z, bias=True, dtype=config.torch_dtype, skip_create_weights=config.skip_create_weights
        )
        self.layer_norm_m = nn.LayerNorm(self.c_m, dtype=config.torch_dtype, eps=config.norm_epsilon)
        self.layer_norm_z = nn.LayerNorm(self.c_z, dtype=config.torch_dtype, eps=config.norm_epsilon)

        bins = torch.linspace(self.min_bin, self.max_bin, self.no_bins)
        squared_bins = bins**2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)
        self.register_buffer("squared_bins", squared_bins)
        self.register_buffer("upper", upper)

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def forward(self, m: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # cast the tensors to the correct dtype
        m = m.to(dtype=self.config.torch_dtype)
        z = z.to(dtype=self.config.torch_dtype)
        x = x.to(dtype=self.config.torch_dtype)

        # [*, N, C_m]
        m_update = self.layer_norm_m(m)

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)

        d = torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True)

        # [*, N, N, no_bins]
        d = ((d > self.squared_bins) * (d < self.upper)).to(x)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = z_update + d

        return m_update, z_update


class ExtraMSAEmbedder(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.linear = Linear(
            config.c_in,
            config.c_out,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        return self.linear(x)


class TemplateSingleEmbedder(nn.Module):
    def __init__(self, c_in: int, c_out: int, dtype: torch.dtype, skip_create_weights: bool = False):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
            dtype:
                Data type of the weights
            skip_create_weights:
                Whether to skip creating weights
        """
        super().__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, dtype=dtype, skip_create_weights=skip_create_weights)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, dtype=dtype, skip_create_weights=skip_create_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, dtype=dtype, skip_create_weights=skip_create_weights)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class TemplateEmbedder(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()

        self.config = config

        mc = config.template_single_embedder
        self.template_single_embedder = TemplateSingleEmbedder(
            c_in=mc.c_in,
            c_out=mc.c_out,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
        )

        mc = config.template_pair_embedder
        self.template_pair_embedder = TemplatePairEmbedder(
            c_in=mc.c_in,
            c_out=mc.c_out,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
        )

        mc = config.template_pair_stack
        self.template_pair_stack = TemplatePairStack(
            c_t=mc.c_t,
            c_hidden_tri_att=mc.c_hidden_tri_att,
            c_hidden_tri_mul=mc.c_hidden_tri_mul,
            no_blocks=mc.no_blocks,
            no_heads=mc.no_heads,
            pair_transition_n=mc.pair_transition_n,
            tri_mul_first=mc.tri_mul_first,
            inf=mc.mask_inf,
            eps=mc.norm_epsilon,
            triangle_attn_backend=mc.triangle_attention_backend,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
        )

        mc = config.template_pointwise_attention
        self.template_pointwise_att = TemplatePointwiseAttention(
            c_t=mc.c_t,
            c_z=mc.c_z,
            c_hidden=mc.c_hidden,
            no_heads=mc.no_heads,
            inf=mc.mask_inf,
            eps=mc.norm_epsilon,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
            chunk_size=mc.chunk_size,
        )
        # Sub-modules may carry independent dtypes; ``forward`` inserts
        # explicit ``.to(dtype=...)`` casts at each module boundary.

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        templ_dim: int,
        skip_template_pair_stack: bool = False,
    ) -> torch.Tensor:
        # Embed the templates one at a time (with a poor man's vmap)
        pair_embeds = []
        z.shape[-2]
        n_templ = batch["template_aatype"].shape[templ_dim]

        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t, idx=idx: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
                batch,
            )

            t = build_template_pair_feat(
                single_template_feats,
                use_unit_vector=self.config.use_unit_vector,
                inf=self.config.mask_inf,
                eps=self.config.norm_epsilon,
                min_bin=self.config.distogram.min_bin,
                max_bin=self.config.distogram.max_bin,
                no_bins=self.config.distogram.no_bins,
            ).to(z)
            t = t.to(dtype=self.config.template_pair_embedder.torch_dtype)
            t = self.template_pair_embedder(t)

            pair_embeds.append(t)

        t_pair = torch.stack(pair_embeds, dim=templ_dim)

        # Cast pair_embedder output to template_pair_stack's dtype — these
        # modules may carry independent dtypes (e.g., pair_embedder fp32 and
        # pair_stack bf16 for the SM80 triangle-attention kernel path).
        t_pair = t_pair.to(dtype=self.config.template_pair_stack.torch_dtype)

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            t_pair,
            pair_mask.unsqueeze(-3).to(t_pair),
            skip_template_pair_stack=skip_template_pair_stack,
        )

        # [*, N, N, C_z]
        desired_dtype = self.config.template_pointwise_attention.torch_dtype
        t = self.template_pointwise_att(
            t.to(dtype=desired_dtype),
            z.to(dtype=desired_dtype),
            template_mask=batch["template_mask"].to(dtype=desired_dtype),
        )

        t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
        # Append singletons
        t_mask = t_mask.reshape(*t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape))))

        t = t * t_mask.to(t)

        ret = {"template_pair_embedding": t}

        if self.config.embed_angles:
            desired_dtype = self.config.template_single_embedder.torch_dtype
            template_angle_feat = build_template_angle_feat(batch)

            # [*, S_t, N, C_m]
            a = self.template_single_embedder(template_angle_feat.to(dtype=desired_dtype))

            ret["template_single_embedding"] = a

        return ret


class TemplatePairEmbedderMultimer(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_dgram: int,
        c_aatype: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        super().__init__()

        self.dgram_linear = Linear(
            c_dgram,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.aatype_linear_1 = Linear(
            c_aatype,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.aatype_linear_2 = Linear(
            c_aatype,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.query_embedding_layer_norm = nn.LayerNorm(c_in, dtype=dtype, eps=eps)
        self.query_embedding_linear = Linear(
            c_in,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.pseudo_beta_mask_linear = Linear(
            1,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.x_linear = Linear(
            1,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.y_linear = Linear(
            1,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.z_linear = Linear(
            1,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.backbone_mask_linear = Linear(
            1,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(
        self,
        template_dgram: torch.Tensor,
        aatype_one_hot: torch.Tensor,
        query_embedding: torch.Tensor,
        pseudo_beta_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        unit_vector: geometry.Vec3Array,
    ) -> torch.Tensor:
        act = 0.0

        pseudo_beta_mask_2d = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act += self.dgram_linear(template_dgram)
        act += self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None])

        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act += self.aatype_linear_1(aatype_one_hot[..., None, :, :])
        act += self.aatype_linear_2(aatype_one_hot[..., None, :])

        backbone_mask_2d = backbone_mask[..., None] * backbone_mask[..., None, :]
        backbone_mask_2d *= multichain_mask_2d
        x, y, z = [(coord * backbone_mask_2d).to(dtype=query_embedding.dtype) for coord in unit_vector]
        act += self.x_linear(x[..., None])
        act += self.y_linear(y[..., None])
        act += self.z_linear(z[..., None])

        act += self.backbone_mask_linear(backbone_mask_2d[..., None].to(dtype=query_embedding.dtype))

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act


class TemplateSingleEmbedderMultimer(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        super().__init__()
        self.template_single_embedder = Linear(
            c_in,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.template_projector = Linear(
            c_out,
            c_out,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        atom_pos: geometry.Vec3Array,
        aatype_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        out = {}

        dtype = batch["template_all_atom_positions"].dtype

        template_chi_angles, template_chi_mask = all_atom_multimer.compute_chi_angles(
            atom_pos,
            batch["template_all_atom_mask"],
            batch["template_aatype"],
        )

        template_features = torch.cat(
            [
                aatype_one_hot,
                torch.sin(template_chi_angles) * template_chi_mask,
                torch.cos(template_chi_angles) * template_chi_mask,
                template_chi_mask,
            ],
            dim=-1,
        ).to(dtype=dtype)

        template_mask = template_chi_mask[..., 0].to(dtype=dtype)

        template_activations = self.template_single_embedder(template_features)
        template_activations = torch.nn.functional.relu(template_activations)
        template_activations = self.template_projector(
            template_activations,
        )

        out["template_single_embedding"] = template_activations
        out["template_mask"] = template_mask

        return out


class TemplateEmbedderMultimer(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()

        self.config = config

        mc = config.template_single_embedder
        self.template_single_embedder = TemplateSingleEmbedderMultimer(
            c_in=mc.c_in,
            c_out=mc.c_out,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
        )

        mc = config.template_pair_embedder
        self.template_pair_embedder = TemplatePairEmbedderMultimer(
            c_in=mc.c_in,
            c_out=mc.c_out,
            c_dgram=mc.c_dgram,
            c_aatype=mc.c_aatype,
            eps=mc.norm_epsilon,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
        )

        mc = config.template_pair_stack
        self.template_pair_stack = TemplatePairStack(
            c_t=mc.c_t,
            c_hidden_tri_att=mc.c_hidden_tri_att,
            c_hidden_tri_mul=mc.c_hidden_tri_mul,
            no_blocks=mc.no_blocks,
            no_heads=mc.no_heads,
            pair_transition_n=mc.pair_transition_n,
            tri_mul_first=mc.tri_mul_first,
            inf=mc.mask_inf,
            eps=mc.norm_epsilon,
            triangle_attn_backend=mc.triangle_attention_backend,
            dtype=mc.torch_dtype,
            skip_create_weights=mc.skip_create_weights,
        )

        self.linear_t = Linear(
            config.c_t,
            config.c_z,
            bias=True,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        padding_mask_2d: torch.Tensor,
        templ_dim: int,
        multichain_mask_2d: torch.Tensor,
        skip_template_pair_stack: bool = False,
    ) -> dict[str, torch.Tensor]:
        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t, idx=idx: torch.index_select(t, templ_dim, idx),
                batch,
            )

            single_template_embeds = {}

            template_positions, pseudo_beta_mask = pseudo_beta_fn(
                single_template_feats["template_aatype"],
                single_template_feats["template_all_atom_positions"],
                single_template_feats["template_all_atom_mask"],
            )

            template_dgram = dgram_from_positions(
                template_positions,
                inf=self.config.mask_inf,
                min_bin=self.config.distogram.min_bin,
                max_bin=self.config.distogram.max_bin,
                no_bins=self.config.distogram.no_bins,
            )

            aatype_one_hot = torch.nn.functional.one_hot(
                single_template_feats["template_aatype"],
                22,
            )

            raw_atom_pos = single_template_feats["template_all_atom_positions"]

            # Vec3Arrays are required to be float32
            atom_pos = geometry.Vec3Array.from_array(raw_atom_pos.to(dtype=torch.float32))

            rigid, backbone_mask = all_atom_multimer.make_backbone_affine(
                atom_pos,
                single_template_feats["template_all_atom_mask"],
                single_template_feats["template_aatype"],
            )
            points = rigid.translation
            rigid_vec = rigid[..., None].inverse().apply_to_point(points)
            unit_vector = rigid_vec.normalized()

            pair_act = self.template_pair_embedder(
                template_dgram,
                aatype_one_hot,
                z,
                pseudo_beta_mask,
                backbone_mask,
                multichain_mask_2d,
                unit_vector,
            )

            single_template_embeds["template_pair_embedding"] = pair_act
            single_template_embeds.update(
                self.template_single_embedder(
                    single_template_feats,
                    atom_pos,
                    aatype_one_hot,
                )
            )
            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            partial(torch.cat, dim=templ_dim),
            template_embeds,
        )

        # Cast pair_embedder output to template_pair_stack's dtype — these
        # modules may carry independent dtypes (e.g., pair_embedder fp32 and
        # pair_stack bf16 for the SM80 triangle-attention kernel path).
        pair_embed = template_embeds["template_pair_embedding"].to(dtype=self.config.template_pair_stack.torch_dtype)

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            pair_embed,
            padding_mask_2d.unsqueeze(-3).to(z),
            skip_template_pair_stack=skip_template_pair_stack,
        )

        # [*, N, N, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        # Cast back to ``linear_t``'s dtype — ``template_pair_stack`` may emit
        # bf16 (independent dtype) while ``linear_t`` inherits the outer
        # TemplateEmbedderMultimer dtype (typically fp32).
        t = t.to(dtype=self.linear_t.weight.dtype)
        t = self.linear_t(t)
        template_embeds["template_pair_embedding"] = t

        return template_embeds
