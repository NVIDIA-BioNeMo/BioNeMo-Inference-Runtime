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
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.transformers.atom import \
    AtomAttentionEncoder
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    BoltzDiffusionTransformer
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.pipeline.boltz.const import (NUM_CHAIN_TYPES,
                                                   NUM_METHOD_TYPES, NUM_TOKENS)


class AtomEmbedding(nn.Module):
    "Boltz1x, Boltz2 Atom Embedding"

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        structure_prediction: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        version: str = "v1",
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            atom_s: int
                The atom single representation dimension.
            atom_z: int
                The atom pair representation dimension.
            token_s: int
                The token single representation dimension.
            token_z: int
                The token pair representation dimension.
            atoms_per_window_queries: int
                The number of atoms per window for queries.
            atoms_per_window_keys: int
                The number of atoms per window for keys.
            atom_feature_dim: int
                The atom feature dimension.
            structure_prediction: bool
                Whether to use structure prediction. Defaults to True.
            use_no_atom_char: bool
                Whether to use no atom character. Defaults to False.
            use_atom_backbone_feat: bool
                Whether to use atom backbone feature. Defaults to False.
            use_residue_feats_atoms: bool
                Whether to use residue feats atoms. Defaults to False.
            version: str = "v1",
                The version. Defaults to "v1".
            eps: float
                The epsilon. Defaults to 1e-5.
            dtype: torch.dtype
                The data type. Defaults to torch.float32.
            mapping: Optional[Mapping]
                The mapping. Defaults to None.
            skip_create_weights: bool
                Whether to skip creating weights. Defaults to False.
        TODO: Implement for the structure prediction.
        """
        super().__init__()
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms
        self.version = version

        self.embed_atom_features = Linear(
            atom_feature_dim,
            atom_s,
            bias=version != "v1",
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)
        self.embed_atompair_ref_pos = Linear(
            3,
            atom_z,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
        )
        self.embed_atompair_ref_dist = Linear(
            1,
            atom_z,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
        )
        self.embed_atompair_mask = Linear(
            1,
            atom_z,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
        )

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            Linear(
                atom_s,
                atom_z,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights,
            ),
        )

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            Linear(
                atom_s,
                atom_z,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights,
            ),
        )

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(
                atom_z,
                atom_z,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights,
            ),
            nn.ReLU(),
            Linear(
                atom_z,
                atom_z,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights,
            ),
            nn.ReLU(),
            Linear(
                atom_z,
                atom_z,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights,
            ),
        )

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(token_s, dtype=dtype, eps=eps),
                Linear(
                    token_s,
                    atom_s,
                    bias=False,
                    dtype=dtype,
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=True,
                    skip_create_weights=skip_create_weights,
                ),
            )

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(token_z, dtype=dtype, eps=eps),
                Linear(
                    token_z,
                    atom_z,
                    bias=False,
                    dtype=dtype,
                    mapping=mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=True,
                    skip_create_weights=skip_create_weights,
                ),
            )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def _compute_atom_feats_v1(self,
                               ref_pos: torch.Tensor,
                               ref_charge: torch.Tensor,
                               atom_pad_mask: torch.Tensor,
                               ref_element: torch.Tensor,
                               ref_atom_name_chars: Optional[
                                   torch.Tensor] = None,
                               **kwargs) -> torch.Tensor:
        """ Compute the atom features for the Boltz1x version. """
        B, N, _ = ref_pos.shape
        atom_feats = torch.cat(
            [
                ref_pos,
                ref_charge.unsqueeze(-1),
                atom_pad_mask.unsqueeze(-1),
                ref_element,
                ref_atom_name_chars.reshape(B, N, 4 * 64),
            ],
            dim=-1,
        )
        return atom_feats

    def _compute_atom_feats_v2(
            self,
            ref_pos: torch.Tensor,
            ref_charge: torch.Tensor,
            ref_element: torch.Tensor,
            ref_atom_name_chars: Optional[torch.Tensor] = None,
            atom_backbone_feat: Optional[torch.Tensor] = None,
            res_type: Optional[torch.Tensor] = None,
            modified: Optional[torch.Tensor] = None,
            mol_type: Optional[torch.Tensor] = None,
            atom_to_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Compute the atom features for the Boltz2 version. """
        B, N, _ = ref_pos.shape
        atom_ref_pos = ref_pos
        atom_feats = [
            atom_ref_pos,
            ref_charge.unsqueeze(-1),
            ref_element,
        ]
        if not self.use_no_atom_char:
            atom_feats.append(ref_atom_name_chars.reshape(B, N, 4 * 64))
        if self.use_atom_backbone_feat:
            atom_feats.append(atom_backbone_feat)
        if self.use_residue_feats_atoms:
            res_feats = torch.cat([
                res_type,
                modified.unsqueeze(-1),
                F.one_hot(mol_type, num_classes=4).float()
            ],
                                  dim=-1)
            atom_to_token = atom_to_token.float()
            atom_res_feats = torch.bmm(atom_to_token, res_feats)
            atom_feats.append(atom_res_feats)
        atom_feats = torch.cat(atom_feats, dim=-1)
        return atom_feats

    def forward(
        self,
        atom_to_token: torch.Tensor,
        ref_pos: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        ref_space_uid: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: Optional[torch.Tensor] = None,
        atom_backbone_feat: Optional[torch.Tensor] = None,
        res_type: Optional[torch.Tensor] = None,
        modified: Optional[torch.Tensor] = None,
        mol_type: Optional[torch.Tensor] = None,
        query_to_keys: Optional[Callable] = None,
        s_trunk: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_to_token: torch.Tensor
                The atom to token mapping. Shape [B, N_atoms, N_tokens]
            ref_pos: torch.Tensor
                The reference positions. Shape [B, N_atoms, 3].
            atom_pad_mask: torch.Tensor
                The atom pad mask. Shape [B, N_atoms].
            ref_space_uid: torch.Tensor
                The reference space uid. Shape [B, N_atoms].
            ref_charge: torch.Tensor
                The reference charge. Shape [B, N_atoms]
            ref_element: torch.Tensor
                The reference element. Shape [B, N_atoms, 128]
            ref_atom_name_chars: Optional[torch.Tensor]
                The reference atom name chars. Shape [B, N_atoms, 4, 64]
            atom_backbone_feat: Optional[torch.Tensor]
                The atom backbone feature. Shape [B, N_atoms, 17]
            res_type: Optional[torch.Tensor]
                The residue type. Shape [B, N_res, 3]
            modified: Optional[torch.Tensor]
                The modified flag. Shape [B, N_res]
            mol_type: Optional[torch.Tensor]
                The mol type. Shape [B, N_res]
            query_to_keys: Optional[Callable]
                The query to keys function.
            s_trunk: Optional[torch.Tensor]
                The s trunk, used for structure prediction. Shape [B, N_tokens, token_s]
            z: Optional[torch.Tensor]
                The z, used for structure prediction. Shape [B, N_tokens, token_z]
        TODO: Support disable or enable torch.autocast.
        """
        B, N, _ = ref_pos.shape
        atom_mask = atom_pad_mask.bool()

        atom_ref_pos = ref_pos
        atom_uid = ref_space_uid

        if self.version == "v1":
            atom_feats = self._compute_atom_feats_v1(
                ref_pos=atom_ref_pos,
                ref_charge=ref_charge,
                atom_pad_mask=atom_pad_mask,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
            )
        elif self.version == "v2":
            atom_feats = self._compute_atom_feats_v2(
                ref_pos=atom_ref_pos,
                ref_charge=ref_charge,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                atom_backbone_feat=atom_backbone_feat,
                res_type=res_type,
                modified=modified,
                mol_type=mol_type,
                atom_to_token=atom_to_token,
            )
        c = self.embed_atom_features(atom_feats)

        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W

        atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
        atom_ref_pos_keys = query_to_keys(atom_ref_pos).view(B, K, 1, H, 3)

        # Float[B, K, W, H, 3]
        d = atom_ref_pos_keys - atom_ref_pos_queries
        # Float[B, K, W, H, 1]
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)

        # AF3 feeds in the reciprocal of the distance norm
        d_norm = 1 / (1 + d_norm)

        atom_mask_queries = atom_mask.view(B, K, W, 1)
        atom_mask_keys = (query_to_keys(atom_mask.unsqueeze(-1).float()).view(
            B, K, 1, H).bool())
        atom_uid_queries = atom_uid.view(B, K, W, 1)
        atom_uid_keys = (query_to_keys(atom_uid.unsqueeze(-1).float()).view(
            B, K, 1, H).long())
        # Bool[B, K, W, H, 1]
        v = ((atom_mask_queries
              & atom_mask_keys
              & (atom_uid_queries == atom_uid_keys)).float().unsqueeze(-1))

        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c

        if self.structure_prediction:
            # run only in structure model not in initial encoding
            atom_to_token = atom_to_token.float()  # Long['b m n'],

            s_to_c = self.s_to_c_trans(s_trunk.float())
            s_to_c = torch.bmm(atom_to_token, s_to_c)
            c = c + s_to_c.to(c)

            atom_to_token_queries = atom_to_token.view(B, K, W,
                                                       atom_to_token.shape[-1])
            atom_to_token_keys = query_to_keys(atom_to_token)
            # squeeze the multiplicity dimension
            atom_to_token_keys = atom_to_token_keys.squeeze(1)
            z_to_p = self.z_to_p_trans(z.float())
            z_to_p = torch.einsum(
                "bijd,bwki,bwlj->bwkld",
                z_to_p,
                atom_to_token_queries,
                atom_to_token_keys,
            )
            p = p + z_to_p.to(p)

        p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(
            query_to_keys(c).view(B, K, 1, H, c.shape[-1]))
        p = p + self.p_mlp(p)

        return q, c, p


class Boltz1InputEmbedder(nn.Module):

    def __init__(self, config: BaseConfig):
        """
        Args:
            config: BaseConfig
                The configuration for the Boltz1InputEmbedder.
        """
        super().__init__()
        self.config = config

        self.atom_embedding = AtomEmbedding(
            atom_s=config.atom_s,
            atom_z=config.atom_z,
            token_s=config.token_s,
            token_z=config.token_z,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            atom_feature_dim=config.atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=False,
            use_atom_backbone_feat=False,
            use_residue_feats_atoms=False,
            version="v1",
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

        # This trick is used to avoid call attention with bias caching.
        self.atom_enc_proj_z = nn.ModuleList()
        diffusion_transformer_config = config.diffusion_transformer
        for _ in range(diffusion_transformer_config.num_blocks):
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(config.atom_z,
                                 dtype=config.torch_dtype,
                                 eps=config.norm_epsilon),
                    Linear(config.atom_z,
                           diffusion_transformer_config.num_heads,
                           bias=False,
                           dtype=config.torch_dtype,
                           mapping=config.mapping,
                           tensor_parallel_mode=TensorParallelMode.COLUMN,
                           gather_output=True,
                           skip_create_weights=config.skip_create_weights),
                ))

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=config.atom_s,
            token_s=config.token_s,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            diffusion_transformer_config=diffusion_transformer_config,
            diffusion_transformer_cls=BoltzDiffusionTransformer,
            structure_prediction=False,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

    def load_weights(self, weights: dict):
        self.atom_embedding.load_weights(weights["atom_embedding"])
        self.atom_attention_encoder.load_weights(
            weights["atom_attention_encoder"])
        for i, proj_z in enumerate(self.atom_enc_proj_z):
            proj_z[0].weight.data.copy_(
                weights[f"atom_enc_proj_z.{i}.0"][0]["weight"])
            proj_z[0].bias.data.copy_(
                weights[f"atom_enc_proj_z.{i}.0"][0]["bias"])
            proj_z[1].load_weights(weights[f"atom_enc_proj_z.{i}.1"])

    def forward(
            self,
            atom_to_token: torch.Tensor,
            ref_pos: torch.Tensor,
            atom_pad_mask: torch.Tensor,
            ref_space_uid: torch.Tensor,
            ref_charge: torch.Tensor,
            ref_element: torch.Tensor,
            ref_atom_name_chars: torch.Tensor,
            res_type: torch.Tensor,
            profile: Optional[torch.Tensor] = None,
            deletion_mean: Optional[torch.Tensor] = None,
            pocket_feature: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token: torch.Tensor
                The atom to token mapping. Shape [B, N_atoms, N_tokens]
            ref_pos: torch.Tensor
                The reference positions. Shape [B, N_atoms, 3].
            atom_pad_mask: torch.Tensor
                The atom pad mask. Shape [B, N_atoms].
            ref_space_uid: torch.Tensor
                The reference space uid. Shape [B, N_atoms].
            ref_charge: torch.Tensor
                The reference charge. Shape [B, N_atoms].
            ref_element: torch.Tensor
                The reference element. Shape [B, N_atoms, 128].
            ref_atom_name_chars: torch.Tensor
                The reference atom name chars. Shape [B, N_atoms, 4, 64].
            res_type: torch.Tensor
                The residue type. Shape [B, N_res, 3].
            profile: torch.Tensor
                The profile. Shape [B, N_res, 17].
            deletion_mean: torch.Tensor
                The deletion mean. Shape [B, N_res, 1].
            pocket_feature: torch.Tensor
                The pocket feature. Shape [B, N_res, 17].
        """
        assert attn_metadata is not None, "Attention metadata is required for Boltz1InputEmbedder"
        assert attn_metadata.query_to_keys is not None, "Query to keys is required for Boltz1InputEmbedder"

        q, c, bias = self.atom_embedding(
            atom_to_token=atom_to_token,
            ref_pos=ref_pos,
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            query_to_keys=attn_metadata.query_to_keys,
        )

        atom_enc_bias = torch.cat(
            [proj_z(bias) for proj_z in self.atom_enc_proj_z], dim=-1)

        # [B, 1, N_res, D]
        a, _, _, = self.atom_attention_encoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            q=q,
            c=c,
            bias=atom_enc_bias,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )

        # multiplicity is 1 for InputEmbedder, we do squeeze here:
        a = a.squeeze(1)
        s = torch.cat(
            [a, res_type, profile,
             deletion_mean.unsqueeze(-1), pocket_feature],
            dim=-1)

        return s


class Boltz2InputEmbedder(nn.Module):

    def __init__(self, config: BaseConfig):
        """
        Args:
            config: BaseConfig
                The configuration for the Boltz2InputEmbedder.
        """
        super().__init__()
        self.config = config

        self.atom_embedding = AtomEmbedding(
            atom_s=config.atom_s,
            atom_z=config.atom_z,
            token_s=config.token_s,
            token_z=config.token_z,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            atom_feature_dim=config.atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=config.use_no_atom_char,
            use_atom_backbone_feat=config.use_atom_backbone_feat,
            use_residue_feats_atoms=config.use_residue_feats_atoms,
            version="v2",
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

        self.atom_enc_proj_z = nn.Sequential(
            nn.LayerNorm(config.atom_z),
            Linear(config.atom_z,
                   config.diffusion_transformer.num_blocks *
                   config.diffusion_transformer.num_heads,
                   bias=False,
                   dtype=config.torch_dtype,
                   mapping=config.mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   gather_output=True,
                   skip_create_weights=config.skip_create_weights),
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=config.atom_s,
            token_s=config.token_s,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            diffusion_transformer_config=config.diffusion_transformer,
            diffusion_transformer_cls=BoltzDiffusionTransformer,
            structure_prediction=False,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

        self.res_type_encoding = Linear(
            NUM_TOKENS,
            config.token_s,
            bias=False,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=config.skip_create_weights)
        self.msa_profile_encoding = Linear(
            NUM_TOKENS + 1,
            config.token_s,
            bias=False,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=config.skip_create_weights)

        self.add_method_conditioning = config.add_method_conditioning
        self.add_modified_flag = config.add_modified_flag
        self.add_cyclic_flag = config.add_cyclic_flag
        self.add_mol_type_feat = config.add_mol_type_feat

        if self.add_method_conditioning:
            self.method_conditioning_init = nn.Embedding(
                NUM_METHOD_TYPES, config.token_s, dtype=config.torch_dtype)
        if self.add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(
                2, config.token_s, dtype=config.torch_dtype)
        if self.add_cyclic_flag:
            self.cyclic_conditioning_init = Linear(
                1,
                config.token_s,
                bias=False,
                dtype=config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=config.skip_create_weights)
        if self.add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                NUM_CHAIN_TYPES, config.token_s)

    def load_weights(self, weights: dict):
        """ Load weights for the Boltz2InputEmbedder """
        self.atom_embedding.load_weights(weights.pop("atom_embedding"))
        self.atom_attention_encoder.load_weights(
            weights.pop("atom_attention_encoder"))

        filter_func = lambda name, _: name.startswith(
            "atom_embedding") or name.startswith("atom_attention_encoder")

        loaded_weight = recursive_calling_load_weights(self, weights,
                                                       filter_func)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
            self,
            atom_to_token: torch.Tensor,
            ref_pos: torch.Tensor,
            atom_pad_mask: torch.Tensor,
            ref_space_uid: torch.Tensor,
            ref_charge: torch.Tensor,
            ref_element: torch.Tensor,
            ref_atom_name_chars: torch.Tensor,
            res_type: torch.Tensor,
            profile: Optional[torch.Tensor] = None,
            deletion_mean: Optional[torch.Tensor] = None,
            pocket_feature: Optional[torch.Tensor] = None,
            atom_backbone_feat: Optional[torch.Tensor] = None,
            method_feature: Optional[torch.Tensor] = None,
            modified: Optional[torch.Tensor] = None,
            cyclic_period: Optional[torch.Tensor] = None,
            mol_type: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token: torch.Tensor
                The atom to token mapping. Shape [B, N_atoms, N_tokens]
            ref_pos: torch.Tensor
                The reference positions. Shape [B, N_atoms, 3].
            atom_pad_mask: torch.Tensor
                The atom pad mask. Shape [B, N_atoms].
            ref_space_uid: torch.Tensor
                The reference space uid. Shape [B, N_atoms].
            ref_charge: torch.Tensor
                The reference charge. Shape [B, N_atoms].
            ref_element: torch.Tensor
                The reference element. Shape [B, N_atoms, 128].
            ref_atom_name_chars: torch.Tensor
                The reference atom name chars. Shape [B, N_atoms, 4, 64].
            res_type: torch.Tensor
                The residue type. Shape [B, N_res, 3].
            profile: torch.Tensor
                The profile. Shape [B, N_res, 17].
            deletion_mean: torch.Tensor
                The deletion mean. Shape [B, N_res, 1].
            pocket_feature: torch.Tensor
                The pocket feature. Shape [B, N_res, 17].
        TODO: Implement for Boltz2Affinity
        """
        assert attn_metadata is not None, "Attention metadata is required for Boltz1InputEmbedder"
        assert attn_metadata.query_to_keys is not None, "Query to keys is required for Boltz1InputEmbedder"

        q, c, bias = self.atom_embedding(
            atom_to_token=atom_to_token,
            ref_pos=ref_pos,
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_backbone_feat=atom_backbone_feat,
            res_type=res_type,
            modified=modified,
            mol_type=mol_type,
            query_to_keys=attn_metadata.query_to_keys,
        )

        atom_enc_bias = self.atom_enc_proj_z(bias)

        # [B, 1, N_res, D]
        a, _, _, = self.atom_attention_encoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            q=q,
            c=c,
            bias=atom_enc_bias,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )

        # multiplicity is 1 for InputEmbedder, we do squeeze here:
        a = a.squeeze(1)
        deletion_mean = deletion_mean.unsqueeze(-1)
        s = (a + self.res_type_encoding(res_type.float()) +
             self.msa_profile_encoding(
                 torch.cat([profile, deletion_mean], dim=-1)))

        if self.add_method_conditioning:
            s = s + self.method_conditioning_init(method_feature)
        if self.add_modified_flag:
            s = s + self.modified_conditioning_init(modified)
        if self.add_cyclic_flag:
            cyclic = cyclic_period.clamp(max=1.0).unsqueeze(-1)
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(mol_type)

        return s
