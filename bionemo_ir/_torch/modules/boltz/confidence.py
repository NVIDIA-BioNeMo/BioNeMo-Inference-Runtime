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

from typing import Any

import torch
from torch import nn

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.layers.conditioning import ContactConditioning
from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.position_encoders import RelativePositionEncoder
from bionemo_ir._torch.utils import recursive_calling_load_weights
from bionemo_ir.configs import BaseConfig
from bionemo_ir.pipeline.models.boltz2.const import (
    bond_types,
    chain_type_ids,
    contact_conditioning_info,
    num_pocket_contact_info,
    num_tokens,
)

from .confidence_utils import (
    compute_aggregated_metric,
    compute_distogram,
    compute_ptms,
    concat_out_dicts,
    repeat_with_multiplicity,
)
from .embedders import Boltz1InputEmbedder
from .trunk import MSAModule, PairformerModule


class Boltz2ConfidenceHeads(nn.Module):
    def __init__(
        self,
        config: BaseConfig = None,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token: int = 23
        self.token_level_confidence = config.token_level_confidence
        self.use_separate_heads = config.use_separate_heads
        self.register_buffer(
            "arange_max_num_atoms", torch.arange(self.max_num_atoms_per_token).reshape(1, 1, -1), persistent=False
        )

        # Weight values for iplddt computation
        self.ligand_weight: int = 20
        self.non_interface_weight: int = 1
        self.interface_weight: int = 10

        if self.use_separate_heads:
            self.to_pae_intra_logits = Linear(
                config.token_z, config.num_pae_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )
            self.to_pae_inter_logits = Linear(
                config.token_z, config.num_pae_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )
        else:
            self.to_pae_logits = Linear(
                config.token_z, config.num_pae_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )

        if self.use_separate_heads:
            self.to_pde_intra_logits = Linear(
                config.token_z, config.num_pde_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )
            self.to_pde_inter_logits = Linear(
                config.token_z, config.num_pde_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )
        else:
            self.to_pde_logits = Linear(
                config.token_z, config.num_pde_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )

        if self.token_level_confidence:
            self.to_plddt_logits = Linear(
                config.token_s, config.num_plddt_bins, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )
            self.to_resolved_logits = Linear(
                config.token_s, 2, bias=False, dtype=dtype, skip_create_weights=skip_create_weights
            )
        else:
            self.to_plddt_logits = Linear(
                config.token_s,
                config.num_plddt_bins * self.max_num_atoms_per_token,
                bias=False,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
            self.to_resolved_logits = Linear(
                config.token_s,
                2 * self.max_num_atoms_per_token,
                bias=False,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )

    def load_weights(self, weights: dict):

        if self.use_separate_heads:
            self.to_pae_intra_logits.load_weights(weights["to_pae_intra_logits"])
            self.to_pae_inter_logits.load_weights(weights["to_pae_inter_logits"])
        else:
            self.to_pae_logits.load_weights(weights["to_pae_logits"])

        if self.use_separate_heads:
            self.to_pde_intra_logits.load_weights(weights["to_pde_intra_logits"])
            self.to_pde_inter_logits.load_weights(weights["to_pde_inter_logits"])
        else:
            self.to_pde_logits.load_weights(weights["to_pde_logits"])

        self.to_resolved_logits.load_weights(weights["to_resolved_logits"])
        self.to_plddt_logits.load_weights(weights["to_plddt_logits"])

    def _compute_pae_outputs(self, z, x_pred, feats, multiplicity, is_same_chain=None, is_different_chain=None):
        """Compute the pae-derived outputs (``pae`` + ptm/iptm/...) in a helper."""
        if self.use_separate_heads:
            pae_intra_logits = self.to_pae_intra_logits(z)
            pae_intra_logits = pae_intra_logits * is_same_chain.float().unsqueeze(-1)

            pae_inter_logits = self.to_pae_inter_logits(z)
            pae_inter_logits = pae_inter_logits * is_different_chain.float().unsqueeze(-1)

            pae_logits = pae_inter_logits + pae_intra_logits
        else:
            pae_logits = self.to_pae_logits(z)

        out: dict[str, torch.Tensor] = {"pae": compute_aggregated_metric(pae_logits, end=32)}
        try:
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(pae_logits, x_pred, feats)
            out["ptm"] = ptm
            out["iptm"] = iptm
            out["ligand_iptm"] = ligand_iptm
            out["protein_iptm"] = protein_iptm
            out["pair_chains_iptm"] = pair_chains_iptm
        except Exception as e:
            print(f"Error in compute_ptms: {e}")
            for _k in ("ptm", "iptm", "ligand_iptm", "protein_iptm", "pair_chains_iptm"):
                out[_k] = z.new_zeros((z.shape[0], z.shape[1]), dtype=torch.float32)
        return out

    def _compute_pde(self, z, is_same_chain=None, is_different_chain=None):
        """Compute the aggregated ``pde`` metric in a helper."""
        z_sym = z + z.transpose(2, 3)
        if self.use_separate_heads:
            pde_intra_logits = self.to_pde_intra_logits(z_sym)
            pde_intra_logits = pde_intra_logits * is_same_chain.float().unsqueeze(-1)

            pde_inter_logits = self.to_pde_inter_logits(z_sym)
            pde_inter_logits = pde_inter_logits * is_different_chain.float().unsqueeze(-1)

            pde_logits = pde_inter_logits + pde_intra_logits
        else:
            pde_logits = self.to_pde_logits(z_sym)
        return compute_aggregated_metric(pde_logits, end=32)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        d: torch.Tensor,
        feats: dict[str, torch.Tensor] | None,
        prob_contact: torch.Tensor,
        multiplicity: int = 1,
    ):
        """
        Args:
            s: torch.Tensor
                s from the confidence module. Shape, [B, mult, N_tokens, token_s].
            z: torch.Tensor
                z from the confidence module. Shape, [B, mult, N_tokens, N_tokens, token_z].
            x_pred: torch.Tensor
                x_pred from the confidence module. Shape, [B, mult, N_atoms, 3].
            d: torch.Tensor
                Pairwise distances between the tokens' representative atoms, from the confidence
                module. Shape, [B, mult, N_tokens, N_tokens].
            feats: Dict[str, torch.Tensor]
                feats from the confidence module (see ``Boltz2ConfidenceModule.forward``).
            prob_contact: torch.Tensor
                Per-pair contact probability, already reduced from the predicted distogram logits by
                ``compute_contact_prob``. Shape, [B, N_tokens, N_tokens].
            multiplicity: int
                multiplicity from the confidence module.
        Returns:
            dict[str, torch.Tensor]
                Output dictionary containing the confidence heads.
        """

        is_same_chain = None
        is_different_chain = None
        if self.use_separate_heads:
            asym_id_token = feats["asym_id"]
            is_same_chain = asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)
            is_same_chain = repeat_with_multiplicity(is_same_chain, multiplicity)
            is_different_chain = ~is_same_chain

        # Compute the pae + pde outputs here, in helpers, so their [N, N, num_bins] logits and the
        # softmax-aggregation temporaries free on return.
        pae_out = self._compute_pae_outputs(z, x_pred, feats, multiplicity, is_same_chain, is_different_chain)
        pde = self._compute_pde(z, is_same_chain, is_different_chain)

        plddt_logits = self.to_plddt_logits(s)

        token_type = feats["mol_type"]

        token_type = repeat_with_multiplicity(token_type, multiplicity)
        is_ligand_token = (token_type == chain_type_ids["NONPOLYMER"]).float()

        assert self.token_level_confidence, "Only support for token level confidence"

        plddt = compute_aggregated_metric(plddt_logits)

        token_pad_mask = repeat_with_multiplicity(feats["token_pad_mask"], multiplicity)

        complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(dim=-1)

        is_contact = (d < 8).float()
        is_different_chain = (feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)).float()

        is_different_chain = repeat_with_multiplicity(is_different_chain, multiplicity)

        token_interface_mask = torch.max(
            is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
            dim=-1,
        ).values
        token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)
        iplddt_weight = (
            is_ligand_token * self.ligand_weight
            + token_interface_mask * self.interface_weight
            + token_non_interface_mask * self.non_interface_weight
        )
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / torch.sum(
            token_pad_mask * iplddt_weight, dim=-1
        )

        # Compute the gPDE and giPDE (pde was aggregated up front in _compute_pde)

        prob_contact = repeat_with_multiplicity(prob_contact, multiplicity)
        token_pad_mask = repeat_with_multiplicity(feats["token_pad_mask"], multiplicity)

        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (1 - torch.eye(token_pad_mask.shape[2], device=token_pad_mask.device).unsqueeze(0).unsqueeze(0))
        )

        token_pair_mask = token_pad_pair_mask * prob_contact

        complex_pde = (pde * token_pair_mask).sum(dim=(2, 3)) / token_pair_mask.sum(dim=(2, 3))

        asym_id = repeat_with_multiplicity(feats["asym_id"], multiplicity)

        token_interface_pair_mask = token_pair_mask * (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2))
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(2, 3)) / (
            token_interface_pair_mask.sum(dim=(2, 3)) + 1e-5
        )

        out_dict = {
            "pde": pde,
            "plddt": plddt,
            "complex_plddt": complex_plddt,
            "complex_iplddt": complex_iplddt,
            "complex_pde": complex_pde,
            "complex_ipde": complex_ipde,
        }
        # pae / ptm outputs were computed up front (see _compute_pae_outputs) so pae_logits freed
        # before this section; merge them in here.
        out_dict.update(pae_out)

        return out_dict


class Boltz2ConfidenceModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        config: BaseConfig = None,
        dtype: torch.dtype | None = None,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.max_num_atoms_per_token = 23
        self.no_update_s = config.no_update_s
        boundaries = torch.linspace(2, config.max_dist, config.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(config.num_dist_bins, config.token_z)

        self.dtype = dtype
        self.skip_create_weights = skip_create_weights

        self.token_level_confidence = config.token_level_confidence
        self.token_s = config.token_s
        self.token_z = config.token_z

        self.s_to_z = Linear(
            self.token_s, self.token_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.s_to_z_transpose = Linear(
            self.token_s, self.token_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.add_s_to_z_prod = config.add_s_to_z_prod
        if self.add_s_to_z_prod:
            self.s_to_z_prod_in1 = Linear(
                self.token_s, self.token_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
            )
            self.s_to_z_prod_in2 = Linear(
                self.token_s, self.token_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
            )
            self.s_to_z_prod_out = Linear(
                self.token_z, self.token_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
            )

        self.s_inputs_norm = nn.LayerNorm(self.token_s, dtype=self.dtype, eps=config.norm_epsilon)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(self.token_s, dtype=self.dtype, eps=config.norm_epsilon)
        self.z_norm = nn.LayerNorm(self.token_z, dtype=self.dtype, eps=config.norm_epsilon)

        self.add_s_input_to_s = config.add_s_input_to_s
        if self.add_s_input_to_s:
            self.s_input_to_s = Linear(
                self.token_s, self.token_s, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
            )

        self.add_z_input_to_z = config.add_z_input_to_z
        if self.add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(
                token_z=self.token_z,
                fix_sym_check=config.fix_sym_check,
                cyclic_pos_enc=config.cyclic_pos_enc,
                period_broadcast=config.relative_position_encoder.period_broadcast,
                dtype=self.dtype,
                skip_create_weights=self.skip_create_weights,
            )
            self.token_bonds = Linear(
                1 if config.maximum_bond_distance == 0 else config.maximum_bond_distance + 2,
                self.token_z,
                bias=False,
                dtype=self.dtype,
                skip_create_weights=self.skip_create_weights,
            )

            self.bond_type_feature = config.bond_type_feature
            if config.bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(bond_types) + 1, config.token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=config.token_z,
                cutoff_min=config.conditioning_cutoff_min,
                cutoff_max=config.conditioning_cutoff_max,
                contact_conditioning_info=contact_conditioning_info,
                dtype=self.dtype,
                skip_create_weights=self.skip_create_weights,
            )

        self.pairformer_stack = PairformerModule(config=config.pairformer)

        self.return_latent_feats = config.return_latent_feats

        self.confidence_heads = Boltz2ConfidenceHeads(
            config=config.confidence_heads, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

    def load_weights(self, weights: dict = None):
        """
        Args:
            weights: The weights of the model. State dict of the original model.
        """
        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weight}")

    def _build_pairformer_inputs(self, x_pred_chunk, s, z, feats, token_to_rep_atom, current_multiplicity):
        """Build ``(d, s_t, z_t, mask, pair_mask)`` for the confidence pairformer.

        Kept in a helper so the large fp32 ``distogram``-embed + ``pair_z`` intermediates (~16 GB at
        large N) are freed on return -- only the (pairformer-dtype) inputs survive into the stack.
        """
        d, distogram = compute_distogram(x_pred_chunk, self.boundaries, token_to_rep_atom, current_multiplicity)
        distogram = self.dist_bin_pairwise_embed(distogram)
        pair_z = repeat_with_multiplicity(z, current_multiplicity) + distogram

        pf_dtype = self.config.pairformer.torch_dtype
        mask = repeat_with_multiplicity(feats["token_pad_mask"], current_multiplicity)
        mask = mask.flatten(0, 1).to(pf_dtype)

        pair_mask = feats["token_pad_mask"][:, :, None] * feats["token_pad_mask"][:, None, :]
        pair_mask = repeat_with_multiplicity(pair_mask, current_multiplicity)
        pair_mask = pair_mask.flatten(0, 1).to(pf_dtype)

        s_t = repeat_with_multiplicity(s, current_multiplicity).flatten(0, 1).to(pf_dtype)
        z_t = pair_z.flatten(0, 1).to(pf_dtype)
        return d, s_t, z_t, mask, pair_mask

    def forward(
        self,
        s_inputs,
        s,
        z,
        x_pred,
        feats,
        prob_contact,
        multiplicity=1,
        max_parallel_samples: int = 1,
        run_sequentially: bool = True,
        attn_metadata: AttentionMetadata | None = None,
    ):
        """
        Inputs:
        s_inputs: (Batch_size, N_tokens, token_s)
        s: (Batch_size, N_tokens, token_s)
        z: (Batch_size, N_tokens, N_tokens, token_z)
        x_pred: (Batch_size * Diffusion_samples, N_atoms, 3) or (Batch_size, Diffusion_samples, N_atoms, 3)
        feats: Dict[str, torch.Tensor]
            - token_bonds:            (Batch_size, N_tokens, N_tokens, 1)
            - token_to_rep_atom:      (Batch_size, N_tokens, N_atoms)
            - token_pad_mask:         (Batch_size, N_tokens)
            - residue_index:          (Batch_size, N_tokens)
            - entity_id:              (Batch_size, N_tokens)
            - cyclic_period:          (Batch_size, N_tokens)
            - token_index:            (Batch_size, N_tokens)
            - sym_id:                 (Batch_size, N_tokens)
            - asym_id:                (Batch_size, N_tokens)
            - type_bonds:             (Batch_size, N_tokens, N_tokens)
            - contact_threshold:      (Batch_size, N_tokens, N_tokens)
            - contact_conditioning:   (Batch_size, N_tokens, N_tokens, len(contact_conditioning_info))
            - mol_type:               (Batch_size, N_tokens)
            - frames_idx:             (Batch_size, N_tokens, 3)
            - atom_to_token:          (Batch_size, N_atoms, N_tokens)
            - atom_pad_mask:          (Batch_size, N_atoms)

        prob_contact: (Batch_size, N_tokens, N_tokens) -- per-pair contact probability, reduced from
            the predicted distogram logits by ``compute_contact_prob`` at the producer.

        Returned Dict[str, torch.Tensor]:
             - pde:                  (Batch_size, Diffusion_samples, N_tokens, N_tokens)
             - plddt:                (Batch_size, Diffusion_samples, N_tokens)
             - complex_plddt:        (Batch_size, Diffusion_samples)
             - complex_iplddt:       (Batch_size, Diffusion_samples)
             - complex_pde:          (Batch_size, Diffusion_samples)
             - complex_ipde:         (Batch_size, Diffusion_samples)
             - pae:                  (Batch_size, Diffusion_samples, N_tokens, N_tokens)
             - ptm:                  (Batch_size, Diffusion_samples)
             - iptm:                 (Batch_size, Diffusion_samples)
             - ligand_iptm:          (Batch_size, Diffusion_samples)
             - protein_iptm:         (Batch_size, Diffusion_samples)
             - pair_chains_iptm:     nested dict, asym_id -> asym_id -> (Batch_size, Diffusion_samples)
             - s_conf, z_conf:       only when ``return_latent_feats``; the pairformer outputs,
                                     (Batch_size, Diffusion_samples, N_tokens, token_s) and
                                     (Batch_size, Diffusion_samples, N_tokens, N_tokens, token_z)
        """
        s = s.to(self.dtype)
        z = z.to(self.dtype)
        if x_pred.ndim == 3:
            BM, N, _ = x_pred.shape
            batch_size = BM // multiplicity
            x_pred = x_pred.reshape(batch_size, multiplicity, *x_pred.shape[1:])
        elif x_pred.ndim == 4:
            batch_size, multiplicity, _, _ = x_pred.shape

        if run_sequentially:
            niter = multiplicity
        else:
            assert max_parallel_samples <= multiplicity, (
                "max_parallel_samples must be less than or equal to multiplicity"
            )
            niter = (multiplicity + max_parallel_samples - 1) // max_parallel_samples

        s_inputs = self.s_inputs_norm(s_inputs)

        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        z = self.z_norm(z)

        if self.add_z_input_to_z:
            relative_position_encoding = self.rel_pos(
                asym_id=feats["asym_id"],
                residue_index=feats["residue_index"],
                entity_id=feats["entity_id"],
                cyclic_period=feats["cyclic_period"],
                token_index=feats["token_index"],
                sym_id=feats["sym_id"],
            )
            # In-place accumulation: z (a freshly-owned z_norm output) is the accumulator, so each
            # [N, N, c_z] add reuses its buffer instead of allocating a new fp32 temporary.
            z += relative_position_encoding
            z += self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z += self.token_bonds_type(feats["type_bonds"].long())
            z += self.contact_conditioning(
                contact_conditioning=feats["contact_conditioning"], contact_threshold=feats["contact_threshold"]
            )

        z += self.s_to_z(s_inputs)[:, :, None, :]
        z += self.s_to_z_transpose(s_inputs)[:, None, :, :]
        if self.add_s_to_z_prod:
            z += self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :] * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        z = z.to(self.config.pairformer.torch_dtype)

        # These raw pair feats are fully consumed by the z-init above (and by the trunk before this
        # module); drop them so their [N, N, *] buffers free before the per-sample pairformer loop.
        for _feat_key in ("contact_conditioning", "contact_threshold", "token_bonds", "type_bonds"):
            feats.pop(_feat_key, None)

        token_to_rep_atom = feats["token_to_rep_atom"]
        out_dicts_chunks = []

        x_chunks = x_pred.chunk(niter, dim=1)
        for x_pred_chunk in x_chunks:
            current_multiplicity = x_pred_chunk.shape[1]
            # Build pairformer inputs in a helper so the fp32 distogram-embed + pair_z (~16 GB)
            # intermediates are freed on return, before the pairformer stack runs.
            d, s_t, z_t, mask, pair_mask = self._build_pairformer_inputs(
                x_pred_chunk, s, z, feats, token_to_rep_atom, current_multiplicity
            )

            s_t, z_t = self.pairformer_stack(s_t, z_t, mask=mask, pair_mask=pair_mask, attn_metadata=attn_metadata)
            s_t = s_t.unflatten(0, (batch_size, -1)).to(self.dtype)
            z_t = z_t.unflatten(0, (batch_size, -1)).to(self.dtype)
            out_dict = {}

            if self.return_latent_feats:
                out_dict["s_conf"] = s_t
                out_dict["z_conf"] = z_t

            out_dict.update(
                self.confidence_heads(
                    s=s_t,
                    z=z_t,
                    x_pred=x_pred_chunk,
                    d=d,
                    feats=feats,
                    multiplicity=current_multiplicity,
                    prob_contact=prob_contact,
                )
            )
            out_dicts_chunks.append(out_dict)

        out_dict = concat_out_dicts(out_dicts_chunks)
        return out_dict


class Boltz1ConfidenceHeads(nn.Module):
    """Confidence heads.
    The compute_pae flag isn't supported in the current implementation.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.token_s = config.token_s
        self.token_z = config.token_z
        self.num_plddt_bins = config.num_plddt_bins
        self.num_pde_bins = config.num_pde_bins
        self.num_pae_bins = config.num_pae_bins

        self.max_num_atoms_per_token = 23
        self.to_pde_logits = Linear(
            self.token_z,
            self.num_pde_bins,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.to_plddt_logits = Linear(
            self.token_s,
            self.num_plddt_bins,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.to_resolved_logits = Linear(
            self.token_s,
            2,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        if self.config.compute_pae:
            self.to_pae_logits = Linear(
                self.token_z,
                self.num_pae_bins,
                bias=False,
                dtype=self.config.torch_dtype,
                skip_create_weights=self.config.skip_create_weights,
            )

    def _compute_pde(self, z):
        """Aggregate ``pde`` in a helper so the ``[N, N, num_pde_bins]`` ``pde_logits`` (+ its
        aggregation temporaries) free on return -- only the small ``[..., N, N]`` pde survives into
        the complex-metric section.
        """
        pde_logits = self.to_pde_logits(z + z.transpose(-3, -2))
        return compute_aggregated_metric(pde_logits, end=32)

    def _compute_pae_outputs(self, z, x_pred, feature_dict):
        """Compute the pae + ptm outputs in a helper so the ``[N, N, num_pae_bins]`` ``pae_logits``
        and its softmax-aggregation temporaries free on return.
        """
        pae_logits = self.to_pae_logits(z)
        out = {"pae": compute_aggregated_metric(pae_logits, end=32)}
        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(pae_logits, x_pred, feature_dict)
        out["ptm"] = ptm
        out["iptm"] = iptm
        out["ligand_iptm"] = ligand_iptm
        out["protein_iptm"] = protein_iptm
        out["pair_chains_iptm"] = pair_chains_iptm
        return out

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        d: torch.Tensor,
        prob_contact: torch.Tensor,
        feature_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            s: torch.Tensor
                s from the confidence module. Shape, [B, mult, N_tokens, token_s].
            z: torch.Tensor
                z from the confidence module. Shape, [B, mult, N_tokens, N_tokens, token_z].
            x_pred: torch.Tensor
                x_pred from the confidence module. Shape, [B, mult, N_atoms, 3].
            d: torch.Tensor
                Pairwise distances between the tokens' representative atoms, from the confidence
                module. Shape, [B, mult, N_tokens, N_tokens].
            prob_contact: torch.Tensor
                Per-pair contact probability, already reduced from the predicted distogram logits by
                ``compute_contact_prob``. Shape, [B, N_tokens, N_tokens].
            feature_dict: dict[str, torch.Tensor]
                feature_dict from the confidence module.
        Return:
            dict[str, torch.Tensor]
                Output dictionary containing scores for the predicted structures.
        """
        B, multiplicity, N_tokens, _ = s.shape
        asym_id = feature_dict["asym_id"]
        token_pad_mask = repeat_with_multiplicity(feature_dict["token_pad_mask"], multiplicity)
        token_type = repeat_with_multiplicity(feature_dict["mol_type"], multiplicity)

        # Compute the pLDDT logits; pde is aggregated in a helper so its [N, N, num_pde_bins]
        # pde_logits frees on return (before the plddt/complex-metric section below).
        plddt_logits = self.to_plddt_logits(s)
        pde = self._compute_pde(z)

        # Weights used to compute the interface pLDDT
        ligand_weight = 2
        interface_weight = 1

        # Retrieve relevant features
        is_ligand_token = (token_type == chain_type_ids["NONPOLYMER"]).float()

        # Compute the aggregated pLDDT and iPLDDT
        plddt = compute_aggregated_metric(plddt_logits)
        complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(dim=-1)

        is_contact = (d < 8).float()
        is_different_chain = (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)).float()
        is_different_chain = repeat_with_multiplicity(is_different_chain, multiplicity)
        token_interface_mask = torch.max(
            is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
            dim=-1,
        ).values
        iplddt_weight = is_ligand_token * ligand_weight + token_interface_mask * interface_weight
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / (
            torch.sum(token_pad_mask * iplddt_weight, dim=-1) + 1e-5
        )

        # Compute the aggregated PDE and iPDE (pde was aggregated up front in _compute_pde)
        prob_contact = repeat_with_multiplicity(prob_contact, multiplicity)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (1 - torch.eye(N_tokens, device=token_pad_mask.device)[None, None, :, :])
        )

        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(dim=(-2, -1)) / token_pair_mask.sum(dim=(-2, -1))
        asym_id = repeat_with_multiplicity(asym_id, multiplicity)
        token_interface_pair_mask = token_pair_mask * (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2))
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(-2, -1)) / (
            token_interface_pair_mask.sum(dim=(-2, -1)) + 1e-5
        )

        out_dict = {
            "pde": pde,
            "plddt": plddt,
            "complex_plddt": complex_plddt,
            "complex_iplddt": complex_iplddt,
            "complex_pde": complex_pde,
            "complex_ipde": complex_ipde,
        }
        if self.config.compute_pae:
            # pae + ptm in a helper so pae_logits + its aggregation temporaries free on return.
            out_dict.update(self._compute_pae_outputs(z, x_pred, feature_dict))
        return out_dict


class Boltz1ConfidenceModule(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.input_embedder_config = config.input_embedder
        self.pairformer_config = config.pairformer
        self.msa_module_config = config.msa_module
        assert self.pairformer_config.torch_dtype == self.msa_module_config.torch_dtype, (
            f"Boltz1ConfidenceModule pairformer dtype: {self.pairformer_config.torch_dtype}, msa dtype: {self.msa_module_config.torch_dtype}"
        )

        self.max_num_atoms_per_token = 23
        self.no_update_s = self.pairformer_config.no_update_s

        self.max_dist = self.config.max_dist
        self.num_dist_bins = self.config.num_dist_bins
        self.token_z = self.config.token_z
        self.token_s = self.config.token_s
        self.s_input_dim = self.token_s + 2 * num_tokens + 1 + num_pocket_contact_info
        boundaries = torch.linspace(2, self.max_dist, self.num_dist_bins - 1)

        self.dist_bin_pairwise_embed = nn.Embedding(self.num_dist_bins, self.token_z)

        self.register_buffer("boundaries", boundaries)

        # Use s diffusion
        self.s_diffusion_norm = nn.LayerNorm(2 * self.token_s)
        self.s_diffusion_to_s = Linear(
            2 * self.token_s,
            self.token_s,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )

        self.s_to_z = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.s_to_z_transpose = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )

        self.add_s_to_z_prod = self.config.add_s_to_z_prod
        if self.add_s_to_z_prod:
            self.s_to_z_prod_in1 = Linear(
                self.s_input_dim,
                self.token_z,
                bias=False,
                dtype=self.config.torch_dtype,
                skip_create_weights=self.config.skip_create_weights,
            )
            self.s_to_z_prod_in2 = Linear(
                self.s_input_dim,
                self.token_z,
                bias=False,
                dtype=self.config.torch_dtype,
                skip_create_weights=self.config.skip_create_weights,
            )
            self.s_to_z_prod_out = Linear(
                self.token_z,
                self.token_z,
                bias=False,
                dtype=self.config.torch_dtype,
                skip_create_weights=self.config.skip_create_weights,
            )

        self.s_init = Linear(
            self.s_input_dim,
            self.token_s,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.z_init_1 = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.z_init_2 = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )

        self.input_embedder = Boltz1InputEmbedder(self.input_embedder_config)
        self.rel_pos = RelativePositionEncoder(
            token_z=self.token_z,
            fix_sym_check=False,
            cyclic_pos_enc=True,
            period_broadcast=True,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.token_bonds = Linear(
            1,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(self.token_s, dtype=self.config.torch_dtype, eps=self.config.norm_epsilon)
        self.z_norm = nn.LayerNorm(self.token_z, dtype=self.config.torch_dtype, eps=self.config.norm_epsilon)

        # Recycling projections
        self.s_recycle = Linear(
            self.token_s,
            self.token_s,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.z_recycle = Linear(
            self.token_z,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            skip_create_weights=self.config.skip_create_weights,
        )

        self.msa_module = MSAModule(self.msa_module_config)
        self.pairformer_module = PairformerModule(self.pairformer_config)

        self.final_s_norm = nn.LayerNorm(self.token_s, dtype=self.config.torch_dtype, eps=self.config.norm_epsilon)
        self.final_z_norm = nn.LayerNorm(self.token_z, dtype=self.config.torch_dtype, eps=self.config.norm_epsilon)

        self.confidence_heads = Boltz1ConfidenceHeads(self.config.heads)

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def get_module_feed_dict(self, feed_dict: dict[str, torch.Tensor], module_name: str) -> dict[str, Any]:
        keys = []
        if module_name == "input_embedder":
            keys = [
                "atom_to_token",
                "ref_pos",
                "atom_pad_mask",
                "ref_space_uid",
                "ref_charge",
                "ref_element",
                "ref_atom_name_chars",
                "res_type",
                "profile",
                "deletion_mean",
                "pocket_feature",
            ]
        elif module_name == "relative_position_encoding":
            keys = ["asym_id", "residue_index", "entity_id", "cyclic_period", "token_index", "sym_id"]
        elif module_name == "msa_module":
            keys = ["msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"]
        else:
            raise ValueError(f"Module name {module_name} not supported")
        return {key: feed_dict.get(key, None) for key in keys}

    def _add_distogram(self, x_chunk, z_chunk, token_to_rep_atom, n_samples):
        """Fold the distogram embedding into ``z_chunk`` in a helper so the large fp32 ``distogram``
        intermediate (~8 GB at large N) frees on return, before the msa_module/pairformer run.
        """
        d, distogram = compute_distogram(
            x_chunk, self.boundaries, token_to_rep_atom, n_samples, dtype=self.config.torch_dtype
        )
        distogram = self.dist_bin_pairwise_embed(distogram)
        z_chunk += distogram  # [B, mult, N_tokens, N_tokens, token_z]
        return d, z_chunk

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        feature_dict: dict[str, torch.Tensor],
        prob_contact: torch.Tensor,
        multiplicity: int = 1,
        s_diffusion: torch.Tensor | None = None,
        max_parallel_samples: int | None = None,
        run_sequentially: bool = True,
        attn_metadata: AttentionMetadata | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the confidence module.
        Args:
            s: torch.Tensor
                s from the trunk module. Shape, [B, N_tokens, token_s].
            z: torch.Tensor
                z from the trunk module. Shape, [B, N_tokens, N_tokens, token_z].
            x_pred: torch.Tensor
                x_pred from the structure module. Shape, [B, mult, N_atoms, 3] or [B*mult, N_atoms, 3].
            feature_dict: dict
                feature_dict from the dataloader.
            prob_contact: torch.Tensor
                Per-pair contact probability, reduced from the distogram module's logits by
                ``compute_contact_prob``. Shape, [B, N_tokens, N_tokens].
            multiplicity: int
                multiplicity from the structure module.
            s_diffusion: Optional[torch.Tensor]
                s_diffusion from the diffusion conditioning module.
            max_parallel_samples: Optional[int]
                max_parallel_samples from the structure module.
            run_sequentially: bool
                Whether to process the confidence samples one at a time, ignoring
                ``max_parallel_samples``.
        return: dict[str, torch.Tensor]
            Output dictionary containing the confidence heads.
        """
        s = s.to(self.config.torch_dtype)
        z = z.to(self.config.torch_dtype)
        if max_parallel_samples is None:
            max_parallel_samples = 1
        if x_pred.ndim == 3:
            B = x_pred.shape[0] // multiplicity  # [B*mult, N_atoms, 3] -> [B, mult, N_atoms, 3]
            x_pred = x_pred.reshape(B, multiplicity, *x_pred.shape[1:])
        else:
            B, multiplicity, _, _ = x_pred.shape

        if run_sequentially:
            niter = multiplicity
        else:
            assert max_parallel_samples <= multiplicity, (
                "max_parallel_samples must be less than or equal to multiplicity"
            )
            niter = (multiplicity + max_parallel_samples - 1) // max_parallel_samples
        x_chunks = x_pred.chunk(niter, dim=1)
        if s_diffusion is not None:
            if s_diffusion.ndim == 3:
                # [B, mult, ...]
                s_diffusion = s_diffusion.reshape(B, multiplicity, *s_diffusion.shape[1:])
            s_diffusion_chunks = s_diffusion.chunk(niter, dim=1)
        else:
            s_diffusion_chunks = [None] * niter

        s_inputs = self.input_embedder(
            **self.get_module_feed_dict(feature_dict, "input_embedder"), attn_metadata=attn_metadata
        )
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]

        relative_position_encoding = self.rel_pos(
            **self.get_module_feed_dict(feature_dict, "relative_position_encoding"),
        )
        # In-place accumulation into the freshly-built z_init / z, avoiding per-term [N, N, c_z]
        # fp32 temporaries during pair init.
        z_init += relative_position_encoding
        z_init += self.token_bonds(feature_dict["token_bonds"].float())

        # Apply recycling
        s = s_init + self.s_recycle(self.s_norm(s))
        z = z_init + self.z_recycle(self.z_norm(z))

        z += self.s_to_z(s_inputs)[:, :, None, :]
        z += self.s_to_z_transpose(s_inputs)[:, None, :, :]

        if self.config.add_s_to_z_prod:
            z += self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :] * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        # token_bonds is fully consumed by the z-init above (and by the trunk before this module);
        # drop it so its [N, N, 1] buffer frees before the per-sample pairformer loop.
        feature_dict.pop("token_bonds", None)

        s = repeat_with_multiplicity(s, multiplicity)
        z = repeat_with_multiplicity(z, multiplicity)
        s_inputs = repeat_with_multiplicity(s_inputs, multiplicity)

        s_chunks = s.chunk(niter, dim=1)
        z_chunks = z.chunk(niter, dim=1)
        s_inputs_chunks = s_inputs.chunk(niter, dim=1)
        out_dicts = []
        for s_inputs_chunk, s_chunk, z_chunk, x_chunk, s_diffusion_chunk in zip(
            s_inputs_chunks, s_chunks, z_chunks, x_chunks, s_diffusion_chunks, strict=True
        ):
            n_samples = x_chunk.shape[1]

            if self.config.use_s_diffusion:
                assert s_diffusion is not None
                s_diffusion_chunk = self.s_diffusion_norm(s_diffusion_chunk)
                s_chunk = s_chunk + self.s_diffusion_to_s(s_diffusion_chunk)
            token_to_rep_atom = feature_dict["token_to_rep_atom"]
            # Fold distogram into z_chunk in a helper so the fp32 distogram embed frees on return,
            # before the msa_module + pairformer_module run.
            d, z_chunk = self._add_distogram(x_chunk, z_chunk, token_to_rep_atom, n_samples)

            mask = feature_dict["token_pad_mask"]
            pair_mask = mask[:, :, None] * mask[:, None, :]  # [B, N_tokens, N_tokens]
            pair_mask = repeat_with_multiplicity(pair_mask, n_samples)  # [B, mult, N_tokens, N_tokens]

            # Currently, MSAModule and Pairformer doesn't support multiplicity > 1, so we reshape here
            # This won't dont change the result for batching
            input_dtype = self.config.pairformer.torch_dtype
            s_chunk = s_chunk.flatten(0, 1).to(input_dtype)
            z_chunk = z_chunk.flatten(0, 1).to(input_dtype)
            s_inputs_chunk = s_inputs_chunk.flatten(0, 1).to(input_dtype)
            mask = repeat_with_multiplicity(mask, n_samples)  # [B, mult, N_tokens]
            mask = mask.flatten(0, 1).to(input_dtype)
            pair_mask = pair_mask.flatten(0, 1).to(input_dtype)

            # FIXME: create attn_metadata for msa_module and pairformer_module
            z_chunk = z_chunk + self.msa_module(
                z=z_chunk,
                emb=s_inputs_chunk,
                token_pad_mask=pair_mask,
                **self.get_module_feed_dict(feature_dict, "msa_module"),
            )

            s_chunk, z_chunk = self.pairformer_module(s=s_chunk, z=z_chunk, mask=mask, pair_mask=pair_mask)

            # Recover dtype for the final output
            s_chunk = s_chunk.to(self.config.torch_dtype)
            z_chunk = z_chunk.to(self.config.torch_dtype)

            s_chunk, z_chunk = self.final_s_norm(s_chunk), self.final_z_norm(z_chunk)

            # Recover multiplicity dimension for heads
            s_chunk = s_chunk.unflatten(0, (B, n_samples))
            z_chunk = z_chunk.unflatten(0, (B, n_samples))

            out_dict = self.confidence_heads(
                s=s_chunk,
                z=z_chunk,
                x_pred=x_chunk,
                d=d,
                feature_dict=feature_dict,
                prob_contact=prob_contact,
            )
            out_dicts.append(out_dict)

        ret = concat_out_dicts(out_dicts)
        return ret
