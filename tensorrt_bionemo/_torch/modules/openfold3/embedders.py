# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

import math
from functools import partial

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_bionemo._torch.modules.openfold2.template import TemplatePairStack
from tensorrt_bionemo._torch.modules.openfold3.sequence_local_atom_attention import AtomAttentionEncoder
from tensorrt_bionemo._torch.modules.openfold3.utils.relpos import relpos_complex
from tensorrt_bionemo._torch.utils import (
    commit_graph_safe_generator,
    make_graph_safe_generator,
    recursive_calling_load_weights,
)
from tensorrt_bionemo.configs.base import BaseConfig


class InputEmbedderAllAtom(nn.Module):
    """
    Embeds a subset of the input features.

    AF3 Algorithm 1 lines 1-5. Includes Algorithms 2 (InputFeatureEmbedder)
    and 3 (RelativePositionEncoding).
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.max_relative_idx = config.max_relative_idx
        self.max_relative_chain = config.max_relative_chain
        self.dtype = config.torch_dtype
        self.skip_create_weights = config.skip_create_weights

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom_ref_element=config.c_atom_ref_element,
            c_atom_ref_name_chars=config.c_atom_ref_name_chars,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token,
            atom_transformer_config=config.atom_transformer_config,
            n_query=config.n_query,
            n_key=config.n_key,
            c_s=config.c_s,
            c_z=config.c_z,
            inf=config.mask_inf,
            eps=config.norm_epsilon,
            add_noisy_pos=config.add_noisy_pos,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
        )

        self.linear_s = Linear(
            config.c_s_input, config.c_s, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.linear_z_ij = Linear(
            config.c_s_input,
            2 * config.c_z,
            bias=False,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )

        num_rel_pos_bins = 2 * self.max_relative_idx + 2
        num_rel_token_bins = 2 * self.max_relative_idx + 2
        num_rel_chain_bins = 2 * self.max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = num_rel_pos_bins + num_rel_token_bins + num_rel_chain_bins + num_same_entity_features

        self.linear_relpos = Linear(
            num_relpos_dims, config.c_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        # Expecting binary feature "token_bonds" of shape [*, N_token, N_token, 1]
        self.linear_token_bonds = Linear(
            1, config.c_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        batch: dict,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary
        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
        """
        # TODO: Check if we need to cast the dtype to float32 here (if accuracy is not affected during inference)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            a, _, _, _ = self.atom_attn_enc(batch=batch, atom_mask=batch["atom_mask"], attn_metadata=attn_metadata)

        a = a.to(dtype=self.linear_s.weight.dtype)

        # [*, N_token, C_s_input]
        s_input = torch.cat(
            [
                a,
                batch["restype"],
                batch["profile"],
                batch["deletion_mean"].unsqueeze(-1),
            ],
            dim=-1,
        )

        # [*, N_token, C_s]
        s = self.linear_s(s_input)

        s_input_emb_ij = self.linear_z_ij(s_input)
        s_input_emb_i, s_input_emb_j = s_input_emb_ij.chunk(2, dim=-1)
        token_bonds_emb = self.linear_token_bonds(batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype))

        # [*, N_token, N_token, C_z]
        z = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]

        relpos_feats = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).to(dtype=z.dtype)
        relpos_emb = self.linear_relpos(relpos_feats)
        z = z + relpos_emb

        z = z + token_bonds_emb

        return s_input, s, z


class MSAModuleEmbedder(nn.Module):
    """Sample MSA features and embed them. Implements AF3 Algorithm 8 lines 1-4.
    This section of the MSAModule is separated from the main stack to allow for
    tensor offloading during inference.
    """

    def __init__(
        self,
        config: BaseConfig,
    ):
        """
        Args:
            c_m_feats:
                MSA input features channel dimension
            c_m:
                MSA channel dimension
            c_s_input:
                Single (s_input) channel dimension
            subsample_main_msa:
                Whether to subsample only the main MSA to a random depth, following
                AF3 SI Section 2.2.
            subsample_all_msa:
                Whether to subsample all MSA (paired + main) to a random depth.
            min_subsampled_all_msa:
                If subsample_all_msa, this specifies the minimum number of MSA
                sequences to retain after subsampling.
            max_subsampled_all_msa:
                If subsample_all_msa, this specifies the minimum number of MSA
                sequences to retain after subsampling.
        """
        super().__init__()

        self.subsample_main_msa = config.subsample_main_msa
        self.subsample_all_msa = config.subsample_all_msa
        self.min_subsampled_all_msa = config.min_subsampled_all_msa
        self.max_subsampled_all_msa = config.max_subsampled_all_msa
        self.dtype = config.torch_dtype
        self.skip_create_weights = config.skip_create_weights
        self.config = config

        self.linear_m = Linear(
            config.c_m_feats, config.c_m, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.linear_s_input = Linear(
            config.c_s_input, config.c_m, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

    @staticmethod
    def _subsample_main_msa(
        msa_feat: torch.Tensor,
        msa_mask: torch.Tensor,
        num_paired_seqs: torch.Tensor,
        asym_id: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample main MSA (unpaired MSA) features for a single sample in the batch.
        The subsampling is independent per each chain.

        Args:
            msa_feat:
                [N_msa, N_token, c_m_feats] MSA features
            msa_mask:
                [N_msa, N_token] Binary mask indicating valid MSA entries.
                The MSA is padded to the maximum MSA dimension across chains,
                so this mask will be all zeros for any chain whose actual MSA dimension
                is less than the maximum.
            num_paired_seqs:
                [] Number of paired MSA sequences
            asym_id:
                [N_token] Id of the chain each token belongs to
        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
                MSA per chain is independently subsampled and re-padded
                to a shared dimension.
        """

        # Set the sequence dimension for the two tensors, the token dimension is this +1
        feat_seq_dim = -3
        mask_seq_dim = -2

        num_paired_seqs = int(num_paired_seqs.item())

        # Separate UniProt paired sequences and main MSA (only the latter is subsampled)
        total_msa_seq = msa_feat.shape[feat_seq_dim]
        num_main_msa_seqs = total_msa_seq - num_paired_seqs

        if num_main_msa_seqs == 0:
            return msa_feat, msa_mask

        split_sections = [num_paired_seqs, num_main_msa_seqs]

        paired_msa_feat, main_msa_feat = torch.split(msa_feat, split_sections, dim=feat_seq_dim)
        paired_msa_mask, main_msa_mask = torch.split(msa_mask, split_sections, dim=mask_seq_dim)

        # Get the length of each chain using consecutive unique asym_id
        _, chain_splits = torch.unique_consecutive(asym_id, return_counts=True)

        # Split the tensor obtaining separate tensors for each chain
        per_chain_msa_feat = torch.split(main_msa_feat, chain_splits.tolist(), dim=feat_seq_dim + 1)
        per_chain_msa_mask = torch.split(main_msa_mask, chain_splits.tolist(), dim=mask_seq_dim + 1)

        # Get the number of main msa seqs per chain
        # summing the ones in the seq dimension in the mask
        # Use float32 as bf16 precision is not enough to distinguish all 16384 integers
        per_chain_main_msa_dim = [
            int(torch.sum(mask, dim=-2, dtype=torch.float32)[..., 0]) for mask in per_chain_msa_mask
        ]

        # Max number of sequences across chains
        max_msa_seqs_across_chains = max(per_chain_main_msa_dim)

        # Dimension to subsample all chains to
        seq_subsample_dim = torch.randint(
            low=1,
            high=int(max_msa_seqs_across_chains + 1),
            size=(1,),
            device=msa_feat.device,
            generator=generator,
        )

        # Get a random permutation of the sequence indexes for each chain
        # Pad it with padding row indexes until max_msa_seqs_across_chains
        chain_index_permutations = [
            torch.cat(
                [
                    torch.randperm(num_seqs, device=msa_feat.device, generator=generator),
                    torch.arange(num_seqs, max_msa_seqs_across_chains, device=msa_feat.device),
                ]
            )[:seq_subsample_dim]
            for num_seqs in per_chain_main_msa_dim
        ]

        # Apply the permutation and keep seq_subsample_dim sequences
        sampled_chain_feats = [
            feat[..., perm, :, :] for feat, perm in zip(per_chain_msa_feat, chain_index_permutations, strict=False)
        ]
        sampled_chain_masks = [
            mask[..., perm, :] for mask, perm in zip(per_chain_msa_mask, chain_index_permutations, strict=False)
        ]

        # Concatenate the chains back together
        sampled_main_msa_feat = torch.cat(sampled_chain_feats, dim=feat_seq_dim + 1)
        sampled_main_msa_mask = torch.cat(sampled_chain_masks, dim=mask_seq_dim + 1)

        # Stack with the uniprot features and mask
        sampled_msa_feat = torch.cat([paired_msa_feat, sampled_main_msa_feat], dim=feat_seq_dim)
        sampled_msa_mask = torch.cat([paired_msa_mask, sampled_main_msa_mask], dim=mask_seq_dim)

        return sampled_msa_feat, sampled_msa_mask

    @staticmethod
    def _subsample_all_msa(
        msa_feat: torch.Tensor,
        msa_mask: torch.Tensor,
        no_subsampled_all_msa: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample all MSA sequences (paired + main) to a fixed number of sequences,
        prioritizing those with at least one non-masked token.

        Args:
            msa_feat:
                [N_msa, N_token, c_m_feats] MSA features
            msa_mask:
                [N_msa, N_token] Binary mask indicating valid MSA entries.
                The MSA is padded to the maximum MSA dimension across chains,
                so this mask will be all zeros for any chain whose actual MSA dimension
                is less than the maximum.
            no_subsampled_all_msa:
                The number of MSA sequences to retain after subsampling.
        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
                MSA per chain is independently subsampled and re-padded
                to a shared dimension.
        """

        # Set the sequence dimension for the two tensors, the token dimension is this +1
        feat_seq_dim = -3
        mask_seq_dim = -2

        if isinstance(no_subsampled_all_msa, torch.Tensor):
            no_subsampled_all_msa = no_subsampled_all_msa.item()

        # Valid msa
        valid_msa = (msa_mask.sum(dim=mask_seq_dim + 1) > 0).squeeze()  # [N_msa]

        if valid_msa.ndim == 0:
            valid_msa = valid_msa.unsqueeze(0)

        valid_idx = valid_msa.nonzero().squeeze()
        invalid_idx = (~valid_msa).nonzero().squeeze()

        if valid_idx.ndim == 0:
            valid_idx = valid_idx.unsqueeze(0)
        if invalid_idx.ndim == 0:
            invalid_idx = invalid_idx.unsqueeze(0)

        device = msa_feat.device
        # Pick msa from the valid ones at random
        if valid_idx.numel() >= no_subsampled_all_msa:
            permuted_idx = valid_idx[torch.randperm(valid_idx.numel(), device=device, generator=generator)]
            selected = permuted_idx[:no_subsampled_all_msa]
        else:
            # Take all valid, then fill with random invalid
            take_invalid = no_subsampled_all_msa - valid_idx.numel()
            if invalid_idx.numel() > 0:
                permuted_idx = invalid_idx[torch.randperm(invalid_idx.numel(), device=device, generator=generator)]
                selected = torch.cat([valid_idx, permuted_idx[:take_invalid]], dim=0)
            else:
                selected = valid_idx

        feat_sub = msa_feat.index_select(feat_seq_dim, selected)
        mask_sub = msa_mask.index_select(mask_seq_dim, selected)
        return feat_sub, mask_sub

    def _apply_subsample_fn_batch(self, fn: callable, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a MSA subsampling function `fn` independently across a batch.
        Contains extra logic to unbind the batch dim prior to sampling
        and pad/stack the output MSA features and mask.
        All arguments are unbound and passed to `fn` per-sample.

        Args:
            fn:
                A function that takes per-sample inputs and returns (feat, mask).
            **kwargs:
               Keyword arguments to forward to `fn`. Assumes each input tensor
               is batched in dim=0.

        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
        """

        batch_size = next(iter(kwargs.values())).shape[0]
        per_sample_kwargs_list = [{k: v[i] for k, v in kwargs.items()} for i in range(batch_size)]

        per_sample_subsampled_msa = []
        per_sample_subsampled_msa_mask = []

        for kwarg in per_sample_kwargs_list:
            subsampled_msa, subsampled_mask = fn(**kwarg)
            per_sample_subsampled_msa.append(subsampled_msa)
            per_sample_subsampled_msa_mask.append(subsampled_mask)

        # Number of sequences to pad to for all the batch
        max_msa_seqs_batch = max([m.shape[-3] for m in per_sample_subsampled_msa])

        def pad_sequences_dim(m, max_seqs, seq_dim):
            """Pad the msa to max_seqs along seq_dim to stack them in a batch"""

            # Add zero padding at start and end for all dimensions after seq_dim
            non_pad_dims = (0, 0) * (abs(seq_dim) - 1)

            # Pad the seq_dim to max_msa_seqs length
            pad = non_pad_dims + (0, max_seqs - m.shape[seq_dim])

            return torch.nn.functional.pad(m, pad)

        # Pad the sequences to same seq length and stack them in a batch
        sampled_msa = torch.stack(
            [pad_sequences_dim(m, max_msa_seqs_batch, seq_dim=-3) for m in per_sample_subsampled_msa],
            dim=0,
        )
        sampled_msa_mask = torch.stack(
            [pad_sequences_dim(m, max_msa_seqs_batch, seq_dim=-2) for m in per_sample_subsampled_msa_mask],
            dim=0,
        )

        return sampled_msa, sampled_msa_mask

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(self, batch: dict, s_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "msa": [*, N_msa, N_token, 32]
                    - "has_deletion": [*, N_msa, N_token]
                    - "deletion_value": [*, N_msa, N_token]
                    - "msa_mask": [*, N_msa, N_token]
                    - "num_paired_seqs": []
                    - "asym_id": [*, N_token]
            s_input:
                [*, N_token, C_s_input] single embedding

        Returns:
            m:
                [*, N_seq, N_token, C_m] MSA embedding
            msa_mask:
                [*, N_seq, N_token] MSA mask
        """
        batch_dims = batch["msa"].shape[:-3]

        # [*, N_msa, N_token, 34]
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        msa_mask = batch["msa_mask"]

        # Draw MSA-subsampling randomness from a private generator so these
        # eager torch.randint / torch.randperm calls stay off the default CUDA
        # generator, which torch.cuda.graph capture of the diffusion module can
        # leave in a graph-registered state (raising "Offset increment outside
        # graph capture"). See make_graph_safe_generator; the default generator
        # is advanced to match afterward so numerics are unchanged.
        subsample = self.subsample_main_msa or self.subsample_all_msa
        generator = make_graph_safe_generator(msa_feat.device) if subsample else None

        if self.subsample_main_msa:
            if math.prod(batch_dims) > 1:
                msa_feat, msa_mask = self._apply_subsample_fn_batch(
                    fn=partial(self._subsample_main_msa, generator=generator),
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    num_paired_seqs=batch["num_paired_seqs"],
                    asym_id=batch["asym_id"],
                )
            else:
                msa_feat, msa_mask = self._subsample_main_msa(
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    num_paired_seqs=batch["num_paired_seqs"],
                    asym_id=batch["asym_id"],
                    generator=generator,
                )
        elif self.subsample_all_msa:
            no_subsampled_all_msa = torch.randint(
                low=self.min_subsampled_all_msa,
                high=int(self.max_subsampled_all_msa + 1),
                size=(1,),
                device=msa_feat.device,
                generator=generator,
            ).item()

            if math.prod(batch_dims) > 1:
                msa_feat, msa_mask = self._apply_subsample_fn_batch(
                    fn=partial(self._subsample_all_msa, generator=generator),
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    no_subsampled_all_msa=torch.full(
                        (msa_feat.shape[0],),
                        no_subsampled_all_msa,
                        device=msa_feat.device,
                    ),
                )
            else:
                msa_feat, msa_mask = self._subsample_all_msa(
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    no_subsampled_all_msa=no_subsampled_all_msa,
                    generator=generator,
                )

        if subsample:
            # Mirror the draws above onto the default generator (numerics
            # unchanged for any downstream RNG consumer).
            commit_graph_safe_generator(generator, msa_feat.device)

        # [*, N_seq, N_token, C_m]
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)

        return m, msa_mask


class TemplatePairEmbedderAllAtom(nn.Module):
    """
    Implements AF3 Algorithm 16 lines 1-5. Also includes line 8.
    The resulting embedded template will go into the TemplatePairStack.
    """

    def __init__(
        self,
        c_in: int,
        c_dgram: int,
        c_aatype: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
        eps: float = 1e-5,
    ):
        """
        Args:
            c_in:
                Pair embedding dimension
            c_out:
                Template pair embedding dimension
            c_dgram:
                Distogram feature embedding dimension
            c_aatype:
                Template aatype feature embedding dimension
            c_out:
                Output channel dimension
        """
        super().__init__()
        self.c_in = c_in
        self.c_dgram = c_dgram
        self.c_aatype = c_aatype
        self.c_out = c_out
        self.dtype = dtype
        self.skip_create_weights = skip_create_weights
        self.eps = eps
        self.dtype = dtype

        # This Feature contains the distogram, pseudo_beta_mask, aatype_1, aatype_2, x, y, z, and backbone mask

        self.template_pair_embedder_merge_feats = Linear(
            self.c_dgram + self.c_aatype * 2 + 5,
            self.c_out,
            bias=False,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_ALL_LINEAR_LAST_DIM),
        )

        self.layer_norm_z = nn.LayerNorm(self.c_in, eps=self.eps, dtype=self.dtype)
        self.linear_z = Linear(
            self.c_in, self.c_out, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

    def _embed_feats(self, batch: dict):
        dtype = batch["template_unit_vector"].dtype

        # [*, N_token, N_token]
        multichain_pair_mask = batch["asym_id"][..., None] == batch["asym_id"][..., None, :]
        multichain_pair_mask = multichain_pair_mask[..., None, :, :, None]

        # [*, N_templ, N_token, N_token]
        pseudo_beta_pair_mask = (
            batch["template_pseudo_beta_mask"][..., None] * batch["template_pseudo_beta_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_distogram = batch["template_distogram"]

        backbone_frame_pair_mask = (
            batch["template_backbone_frame_mask"][..., None] * batch["template_backbone_frame_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_unit_vector = batch["template_unit_vector"]
        x, y, z = template_unit_vector.unbind(dim=-1)

        # [*, N_templ, N_token, N_token, 32]
        template_restype = batch["template_restype"]
        n_token = batch["template_restype"].shape[-2]
        template_restype_ti = template_restype[..., None, :].expand(*template_restype.shape[:-2], -1, n_token, -1)
        template_restype_tj = template_restype[..., None, :, :].expand(*template_restype.shape[:-2], n_token, -1, -1)

        a = torch.cat(
            [
                template_distogram,
                pseudo_beta_pair_mask,
                template_restype_ti.to(dtype=dtype),
                template_restype_tj.to(dtype=dtype),
                x[..., None],
                y[..., None],
                z[..., None],
                backbone_frame_pair_mask,
            ],
            dim=-1,
        )
        a = self.template_pair_embedder_merge_feats(a)

        return a

    def forward(self, batch, z):
        """
        Args:
            batch:
                Input template feature dictionary
            z:
                Pair embedding
        Returns:
            # [*, N_templ, N_token, N_token, C_out] Template pair feature embedding
        """
        a = self._embed_feats(batch=batch)

        # [*, N_templ, N_token, N_token, C_out]
        z = self.linear_z(self.layer_norm_z(z))
        z = z[..., None, :, :, :] + a

        return z


class TemplateEmbedderAllAtom(nn.Module):
    """Implements AF3 Algorithm 16."""

    def __init__(self, config: BaseConfig):
        """
        Args:
            config:
                ConfigDict with template config.
        """
        super().__init__()

        self.dtype = config.torch_dtype
        self.skip_create_weights = config.skip_create_weights
        self.eps = config.norm_epsilon
        self.inf = config.mask_inf
        self.config = config

        self.template_pair_embedder = TemplatePairEmbedderAllAtom(
            c_in=config.template_pair_embedder.c_in,
            c_dgram=config.template_pair_embedder.c_dgram,
            c_aatype=config.template_pair_embedder.c_aatype,
            c_out=config.template_pair_embedder.c_out,
            dtype=config.torch_dtype,
            skip_create_weights=config.skip_create_weights,
            eps=self.eps,
        )

        tri_mul_keys = ["p_in", "g_in", "p_out", "g_out"]
        tri_attn_keys = ["q", "k", "v", "g", "z", "o"]

        tri_mul_out_bias = dict.fromkeys(tri_mul_keys, False)
        tri_mul_in_bias = dict.fromkeys(tri_mul_keys, False)
        tri_attn_start_bias = dict.fromkeys(tri_attn_keys, False)
        tri_attn_end_bias = dict.fromkeys(tri_attn_keys, False)

        self.template_pair_stack = TemplatePairStack(
            c_t=config.template_pair_stack.c_t,
            c_hidden_tri_att=config.template_pair_stack.c_hidden_tri_att,
            c_hidden_tri_mul=config.template_pair_stack.c_hidden_tri_mul,
            no_blocks=config.template_pair_stack.no_blocks,
            no_heads=config.template_pair_stack.no_heads,
            pair_transition_n=config.template_pair_stack.pair_transition_n,
            tri_mul_first=config.template_pair_stack.tri_mul_first,
            trimul_high_precision=config.template_pair_stack.trimul_high_precision,
            triangle_attn_backend=config.triangle_attention_backend,
            transition_type=config.template_pair_stack.transition_type,
            tri_mul_out_bias=tri_mul_out_bias,
            tri_mul_in_bias=tri_mul_in_bias,
            tri_attn_start_bias=tri_attn_start_bias,
            tri_attn_end_bias=tri_attn_end_bias,
            inf=self.inf,
            dtype=config.template_pair_stack.torch_dtype,
            skip_create_weights=self.skip_create_weights,
            eps=self.eps,
        )

        self.linear_t = Linear(
            config.template_pair_stack.c_t,
            config.c_z,
            bias=False,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(self, batch: dict, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary
            z:
                [*, N_token, N_token, C_z] Pair embedding
            pair_mask:
                [*, N_token, N_token] Pair mask

        Returns:
            t:
                [*, N_token, N_token, C_z] Template embedding
        """

        # [*, N_templ, N_token, N_token, C_t]
        template_embeds = self.template_pair_embedder(batch, z)

        n_templ = template_embeds.shape[-4]

        # [*, 1, N_token, N_token]
        pair_mask = pair_mask[..., None, :, :].to(dtype=z.dtype)

        # [*, N_templ, N_token, N_token, C_z]
        # The template pair stack may use CuTeDSL triangle attention which
        # requires bf16/fp16.  Cast inputs to the stack's compute dtype when
        # the embedder output is wider (e.g. fp32), then cast back afterwards.
        embed_dtype = template_embeds.dtype
        stack_block = self.template_pair_stack.blocks[0]
        stack_dtype = getattr(stack_block, "dtype", embed_dtype)
        if embed_dtype != stack_dtype:
            template_embeds = template_embeds.to(dtype=stack_dtype)
            pair_mask = pair_mask.to(dtype=stack_dtype)
        t = self.template_pair_stack(t=template_embeds, mask=pair_mask)
        if t.dtype != embed_dtype:
            t = t.to(dtype=embed_dtype)

        # [*, N_token, N_token, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        t = self.linear_t(t)

        return t
