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

from typing import Any, Dict, Optional

import torch
from tensorrt_llm.functional import AllReduceParams
from torch import nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.conditioning import ContactConditioning
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import \
    RelativePositionEncoder
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.pipeline.boltz.const import (BOND_TYPES, CHAIN_TYPE_IDS,
                                                   CONTACT_CONDITIONING_INFO,
                                                   NUM_POCKET_CONTACT_INFO,
                                                   NUM_TOKENS)

from .confidence_utils import (compute_aggregated_metric, compute_distogram,
                               compute_ptms, concat_out_dicts,
                               repeat_with_multiplicity)
from .embedders import Boltz1InputEmbedder
from .trunk import MSAModule, PairformerModule


class Boltz2ConfidenceHeads(nn.Module):

    def __init__(
        self,
        config: BaseConfig = None,
        dtype: torch.dtype = torch.float32,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token: int = 23
        self.token_level_confidence = config.token_level_confidence
        self.use_separate_heads = config.use_separate_heads
        contacts = torch.zeros((1, 1, 1, 1, 64), dtype=dtype)
        contacts[:, :, :, :, :20] = 1.0

        self.register_buffer("contacts", contacts, persistent=False)

        self.register_buffer('arange_max_num_atoms',
                             torch.arange(self.max_num_atoms_per_token).reshape(
                                 1, 1, -1),
                             persistent=False)

        # Weight values for iplddt computation
        self.ligand_weight: int = 20
        self.non_interface_weight: int = 1
        self.interface_weight: int = 10

        if self.use_separate_heads:
            self.to_pae_intra_logits = Linear(
                config.token_z,
                config.num_pae_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
            self.to_pae_inter_logits = Linear(
                config.token_z,
                config.num_pae_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
        else:
            self.to_pae_logits = Linear(
                config.token_z,
                config.num_pae_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)

        if self.use_separate_heads:
            self.to_pde_intra_logits = Linear(
                config.token_z,
                config.num_pde_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
            self.to_pde_inter_logits = Linear(
                config.token_z,
                config.num_pde_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
        else:
            self.to_pde_logits = Linear(
                config.token_z,
                config.num_pde_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)

        if self.token_level_confidence:
            self.to_plddt_logits = Linear(
                config.token_s,
                config.num_plddt_bins,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
            self.to_resolved_logits = Linear(
                config.token_s,
                2,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
        else:
            self.to_plddt_logits = Linear(
                config.token_s,
                config.num_plddt_bins * self.max_num_atoms_per_token,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
            self.to_resolved_logits = Linear(
                config.token_s,
                2 * self.max_num_atoms_per_token,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)

    def load_weights(self, weights: dict):

        if self.use_separate_heads:
            self.to_pae_intra_logits.load_weights(
                weights["to_pae_intra_logits"])
            self.to_pae_inter_logits.load_weights(
                weights["to_pae_inter_logits"])
        else:
            self.to_pae_logits.load_weights(weights["to_pae_logits"])

        if self.use_separate_heads:
            self.to_pde_intra_logits.load_weights(
                weights["to_pde_intra_logits"])
            self.to_pde_inter_logits.load_weights(
                weights["to_pde_inter_logits"])
        else:
            self.to_pde_logits.load_weights(weights["to_pde_logits"])

        self.to_resolved_logits.load_weights(weights["to_resolved_logits"])
        self.to_plddt_logits.load_weights(weights["to_plddt_logits"])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        d: torch.Tensor,
        feats: Optional[Dict[str, torch.Tensor]],
        pred_distogram_logits: torch.Tensor,
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
                d from the confidence module. Shape, [B, mult, N_atoms, N_atoms].
            feats: Dict[str, torch.Tensor]
                feats from the confidence module. Shape, [B, mult, N_tokens, N_atoms].
            pred_distogram_logits: torch.Tensor
                pred_distogram_logits from the confidence module. Shape, [B, mult, N_atoms, N_atoms, num_dist_bins].
            multiplicity: int
                multiplicity from the confidence module.
        Returns:
            dict[str, torch.Tensor]
                Output dictionary containing the confidence heads.
        """

        if self.use_separate_heads:
            asym_id_token = feats["asym_id"]
            is_same_chain = asym_id_token.unsqueeze(
                -1) == asym_id_token.unsqueeze(-2)
            is_same_chain = repeat_with_multiplicity(is_same_chain,
                                                     multiplicity)
            is_different_chain = ~is_same_chain

        if self.use_separate_heads:
            pae_intra_logits = self.to_pae_intra_logits(z)
            pae_intra_logits = pae_intra_logits * is_same_chain.float(
            ).unsqueeze(-1)

            pae_inter_logits = self.to_pae_inter_logits(z)
            pae_inter_logits = pae_inter_logits * is_different_chain.float(
            ).unsqueeze(-1)

            pae_logits = pae_inter_logits + pae_intra_logits
        else:
            pae_logits = self.to_pae_logits(z)

        if self.use_separate_heads:
            pde_intra_logits = self.to_pde_intra_logits(z + z.transpose(2, 3))
            pde_intra_logits = pde_intra_logits * is_same_chain.float(
            ).unsqueeze(-1)

            pde_inter_logits = self.to_pde_inter_logits(z + z.transpose(2, 3))
            pde_inter_logits = pde_inter_logits * is_different_chain.float(
            ).unsqueeze(-1)

            pde_logits = pde_inter_logits + pde_intra_logits
        else:
            pde_logits = self.to_pde_logits(z + z.transpose(2, 3))

        resolved_logits = self.to_resolved_logits(s)
        plddt_logits = self.to_plddt_logits(s)

        token_type = feats["mol_type"]

        token_type = repeat_with_multiplicity(token_type, multiplicity)
        is_ligand_token = (token_type == CHAIN_TYPE_IDS["NONPOLYMER"]).float()

        assert self.token_level_confidence, "Only support for token level confidence"

        plddt = compute_aggregated_metric(plddt_logits)

        token_pad_mask = repeat_with_multiplicity(feats["token_pad_mask"],
                                                  multiplicity)

        complex_plddt = (plddt * token_pad_mask).sum(
            dim=-1) / token_pad_mask.sum(dim=-1)

        is_contact = (d < 8).float()
        is_different_chain = (feats["asym_id"].unsqueeze(-1)
                              != feats["asym_id"].unsqueeze(-2)).float()

        is_different_chain = repeat_with_multiplicity(is_different_chain,
                                                      multiplicity)

        token_interface_mask = torch.max(
            is_contact * is_different_chain *
            (1 - is_ligand_token).unsqueeze(-1),
            dim=-1,
        ).values
        token_non_interface_mask = (1 - token_interface_mask) * (
            1 - is_ligand_token)
        iplddt_weight = (is_ligand_token * self.ligand_weight +
                         token_interface_mask * self.interface_weight +
                         token_non_interface_mask * self.non_interface_weight)
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(
            dim=-1) / torch.sum(token_pad_mask * iplddt_weight, dim=-1)

        # Compute the gPDE and giPDE

        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = repeat_with_multiplicity(
            nn.functional.softmax(pred_distogram_logits, dim=-1), multiplicity)

        prob_contact = (pred_distogram_prob * self.contacts).sum(-1)
        token_pad_mask = repeat_with_multiplicity(feats["token_pad_mask"],
                                                  multiplicity)

        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1) * token_pad_mask.unsqueeze(-2) *
            (1 -
             torch.eye(token_pad_mask.shape[2],
                       device=token_pad_mask.device).unsqueeze(0).unsqueeze(0)))

        token_pair_mask = token_pad_pair_mask * prob_contact

        complex_pde = (pde * token_pair_mask).sum(
            dim=(2, 3)) / token_pair_mask.sum(dim=(2, 3))

        asym_id = repeat_with_multiplicity(feats["asym_id"], multiplicity)

        token_interface_pair_mask = token_pair_mask * (asym_id.unsqueeze(-1)
                                                       != asym_id.unsqueeze(-2))
        complex_ipde = (pde * token_interface_pair_mask).sum(
            dim=(2, 3)) / (token_interface_pair_mask.sum(dim=(2, 3)) + 1e-5)

        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
        )
        out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

        try:
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
                pae_logits, x_pred, feats)
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm
        except Exception as e:
            print(f"Error in compute_ptms: {e}")
            out_dict["ptm"] = torch.zeros_like(complex_plddt)
            out_dict["iptm"] = torch.zeros_like(complex_plddt)
            out_dict["ligand_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["protein_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["pair_chains_iptm"] = torch.zeros_like(complex_plddt)

        return out_dict


class Boltz2ConfidenceModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        config: BaseConfig = None,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.max_num_atoms_per_token = 23
        self.no_update_s = config.no_update_s
        boundaries = torch.linspace(2, config.max_dist,
                                    config.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(config.num_dist_bins,
                                                    config.token_z)

        self.dtype = dtype
        self.mapping = mapping
        self.skip_create_weights = skip_create_weights

        self.token_level_confidence = config.token_level_confidence
        self.token_s = config.token_s
        self.token_z = config.token_z

        self.s_to_z = Linear(self.token_s,
                             self.token_z,
                             bias=False,
                             dtype=self.dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=self.skip_create_weights)

        self.s_to_z_transpose = Linear(
            self.token_s,
            self.token_z,
            bias=False,
            dtype=self.dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.skip_create_weights)

        self.add_s_to_z_prod = config.add_s_to_z_prod
        if self.add_s_to_z_prod:
            self.s_to_z_prod_in1 = Linear(
                self.token_s,
                self.token_z,
                bias=False,
                dtype=self.dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.skip_create_weights)
            self.s_to_z_prod_in2 = Linear(
                self.token_s,
                self.token_z,
                bias=False,
                dtype=self.dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.skip_create_weights)
            self.s_to_z_prod_out = Linear(
                self.token_z,
                self.token_z,
                bias=False,
                dtype=self.dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.skip_create_weights)

        self.s_inputs_norm = nn.LayerNorm(self.token_s,
                                          dtype=self.dtype,
                                          eps=config.norm_epsilon)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(self.token_s,
                                       dtype=self.dtype,
                                       eps=config.norm_epsilon)
        self.z_norm = nn.LayerNorm(self.token_z,
                                   dtype=self.dtype,
                                   eps=config.norm_epsilon)

        self.add_s_input_to_s = config.add_s_input_to_s
        if self.add_s_input_to_s:
            self.s_input_to_s = Linear(
                self.token_s,
                self.token_s,
                bias=False,
                dtype=self.dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.skip_create_weights)

        self.add_z_input_to_z = config.add_z_input_to_z
        if self.add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(
                token_z=self.token_z,
                fix_sym_check=config.fix_sym_check,
                cyclic_pos_enc=config.cyclic_pos_enc,
                period_broadcast=config.relative_position_encoder.
                period_broadcast,
                dtype=self.dtype,
                mapping=self.mapping,
                skip_create_weights=self.skip_create_weights)
            self.token_bonds = Linear(
                1 if config.maximum_bond_distance == 0 else
                config.maximum_bond_distance + 2,
                self.token_z,
                bias=False,
                dtype=self.dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.skip_create_weights)

            self.bond_type_feature = config.bond_type_feature
            if config.bond_type_feature:
                self.token_bonds_type = nn.Embedding(
                    len(BOND_TYPES) + 1, config.token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=config.token_z,
                cutoff_min=config.conditioning_cutoff_min,
                cutoff_max=config.conditioning_cutoff_max,
                contact_conditioning_info=CONTACT_CONDITIONING_INFO,
                dtype=self.dtype,
                mapping=self.mapping,
                skip_create_weights=self.skip_create_weights)

        self.pairformer_stack = PairformerModule(config=config.pairformer)

        self.return_latent_feats = config.return_latent_feats

        self.confidence_heads = Boltz2ConfidenceHeads(
            config=config.confidence_heads,
            dtype=self.dtype,
            mapping=self.mapping,
            skip_create_weights=self.skip_create_weights)

    def load_weights(self, weights: dict = None):
        """
        Args:
            weights: The weights of the model. State dict of the original model.
        """
        self.pairformer_stack.load_weights(weights.pop("pairformer"))
        self.confidence_heads.load_weights(weights.pop("confidence_heads"))

        if self.add_z_input_to_z:
            self.rel_pos.linear.load_weights(weights.pop("rel_pos"))
            self.contact_conditioning.load_weights(
                weights.pop("contact_conditioning"))

        filter_func = lambda name, _: name.startswith("rel_pos") or \
                         name.startswith("pairformer") or \
                         name.startswith("confidence_heads") or \
                         name.startswith("contact_conditioning")

        loaded_weight = recursive_calling_load_weights(self, weights,
                                                       filter_func)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weight}")

    def forward(self,
                s_inputs,
                s,
                z,
                x_pred,
                feats,
                pred_distogram_logits,
                multiplicity=1,
                max_parallel_samples: int = 1,
                run_sequentially=True,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        """
        Inputs:
        s_inputs: (Batch_size, N_atoms, token_s)
        s: (Batch_size, N_atoms, token_s)
        z: (Batch_size, N_atoms, N_atoms, token_z)
        x_pred: (Batch_size * Diffusion_samples, N, 3) or (Batch_size, Diffusion_samples, N, 3)
        feats: Dict[str, torch.Tensor]
            - token_bonds:            (Batch_size, N_atoms, N_atoms, 1)
            - token_to_rep_atom:      (Batch_size, N_atoms, N)
            - token_pad_mask:         (Batch_size, N_atoms)
            - residue_index:          (Batch_size, N_atoms)
            - entity_id:              (Batch_size, N_atoms)
            - cyclic_period:          (Batch_size, N_atoms)
            - token_index:            (Batch_size, N_atoms)
            - sym_id:                 (Batch_size, N_atoms)
            - asym_id:                (Batch_size, N_atoms)
            - type_bonds:             (Batch_size, N_atoms, N_atoms)
            - contact_threshold:      (Batch_size, N_atoms, N_atoms)
            - contact_conditioning:   (Batch_size, N_atoms, N_atoms, 5)
            - mol_type:               (Batch_size, N_atoms)
            - frames_idx:             (Batch_size, N_atoms, 3)
            - atom_to_token:          (Batch_size, N, N_atoms)
            - atom_pad_mask:          (Batch_size, N)

        pred_distogram_logits: [Batch_size, N_atoms, N_atoms, num_dist_bins]

        Returned Dict[str, torch.Tensor]:
             - pde_logits:           (Batch_size, Diffusion_samples, N_atoms, N_atoms, num_pde_bins)
             - plddt_logits:         (Batch_size, Diffusion_samples, N_atoms, num_plddt_bins)
             - resolved_logits:      (Batch_size, Diffusion_samples, N_atoms, num_resolved_bins)
             - pde:                  (Batch_size, Diffusion_samples, N_atoms, N_atoms)
             - plddt:                (Batch_size, Diffusion_samples, N_atoms)
             - complex_plddt:        (Batch_size, Diffusion_samples)
             - complex_iplddt:       (Batch_size, Diffusion_samples)
             - complex_pde:          (Batch_size, Diffusion_samples)
             - complex_ipde:         (Batch_size, Diffusion_samples)
             - pae_logits:           (Batch_size, Diffusion_samples, N_atoms, N_atoms, num_pae_bins)
             - pae:                  (Batch_size, Diffusion_samples, N_atoms, N_atoms)
             - ptm:                  (Batch_size, Diffusion_samples)
             - iptm:                 (Batch_size, Diffusion_samples)
             - ligand_iptm:          (Batch_size, Diffusion_samples)
             - protein_iptm:         (Batch_size, Diffusion_samples)
             - pair_chains_iptm:     dict("0": (Batch_size, Diffusion_samples), "1": (Batch_size, Diffusion_samples), ...)
        """
        s = s.to(self.dtype)
        z = z.to(self.dtype)
        if x_pred.ndim == 3:
            BM, N, _ = x_pred.shape
            batch_size = BM // multiplicity
            x_pred = x_pred.reshape(batch_size, multiplicity, *x_pred.shape[1:])
        elif x_pred.ndim == 4:
            batch_size, multiplicity, _, _ = x_pred.shape

        niter = (multiplicity + max_parallel_samples -
                 1) // max_parallel_samples

        assert max_parallel_samples <= multiplicity, "max_parallel_samples must be less than or equal to multiplicity"

        if not run_sequentially:
            max_parallel_samples = multiplicity

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
            z = z + relative_position_encoding
            z = z + self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z = z + self.token_bonds_type(feats["type_bonds"].long())
            z = z + self.contact_conditioning(
                contact_conditioning=feats["contact_conditioning"],
                contact_threshold=feats["contact_threshold"])

        z = (z + self.s_to_z(s_inputs)[:, :, None, :] +
             self.s_to_z_transpose(s_inputs)[:, None, :, :])
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :] *
                self.s_to_z_prod_in2(s_inputs)[:, None, :, :])

        token_to_rep_atom = feats["token_to_rep_atom"]
        out_dicts_chunks = []

        x_chunks = x_pred.chunk(niter, dim=1)
        for x_pred_chunk in x_chunks:
            current_multiplicity = x_pred_chunk.shape[1]
            d, distogram = compute_distogram(x_pred_chunk, self.boundaries,
                                             token_to_rep_atom,
                                             current_multiplicity)
            distogram = self.dist_bin_pairwise_embed(distogram)
            pair_z = repeat_with_multiplicity(z,
                                              current_multiplicity) + distogram

            mask = repeat_with_multiplicity(feats["token_pad_mask"],
                                            current_multiplicity)
            mask = mask.flatten(0, 1).to(self.config.pairformer.torch_dtype)

            pair_mask = feats["token_pad_mask"][:, :, None] * feats[
                "token_pad_mask"][:, None, :]
            pair_mask = repeat_with_multiplicity(pair_mask,
                                                 current_multiplicity)
            pair_mask = pair_mask.flatten(0, 1).to(
                self.config.pairformer.torch_dtype)

            s_t = repeat_with_multiplicity(s, current_multiplicity).flatten(
                0, 1).to(self.config.pairformer.torch_dtype)
            z_t = pair_z.flatten(0, 1).to(self.config.pairformer.torch_dtype)

            s_t, z_t = self.pairformer_stack(
                s_t,
                z_t,
                mask=mask,
                pair_mask=pair_mask,
                attn_metadata=attn_metadata,
                all_reduce_params=all_reduce_params)
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
                    pred_distogram_logits=pred_distogram_logits,
                ))
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

        contacts = torch.zeros((1, 1, 1, 1, 64), dtype=self.config.torch_dtype)
        contacts[:, :, :, :, :20] = 1.0

        self.register_buffer("contacts", contacts, persistent=False)

        self.max_num_atoms_per_token = 23
        self.to_pde_logits = Linear(
            self.token_z,
            self.num_pde_bins,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        self.to_plddt_logits = Linear(
            self.token_s,
            self.num_plddt_bins,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        self.to_resolved_logits = Linear(
            self.token_s,
            2,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        if self.config.compute_pae:
            self.to_pae_logits = Linear(
                self.token_z,
                self.num_pae_bins,
                bias=False,
                dtype=self.config.torch_dtype,
                mapping=self.config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.config.skip_create_weights)

    def load_weights(self, weights: dict) -> None:
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        d: torch.Tensor,
        pred_distogram_logits: torch.Tensor,
        feature_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            s: torch.Tensor
                s from the confidence module. Shape, [B, mult, N_tokens, token_s].
            z: torch.Tensor
                z from the confidence module. Shape, [B, mult, N_tokens, N_tokens, token_z].
            pred_distogram_logits: torch.Tensor
                pred_distogram_logits from the confidence module. Shape, [B, N_tokens, N_tokens, num_dist_bins].
        Return:
            dict[str, torch.Tensor]
                Output dictionary containing scores for the predicted structures.
        """
        B, multiplicity, N_tokens, _ = s.shape
        asym_id = feature_dict["asym_id"]
        token_pad_mask = repeat_with_multiplicity(
            feature_dict["token_pad_mask"], multiplicity)
        token_type = repeat_with_multiplicity(feature_dict["mol_type"],
                                              multiplicity)

        # Compute the pLDDT, PDE, PAE, and resolved logits
        plddt_logits = self.to_plddt_logits(s)
        pde_logits = self.to_pde_logits(z + z.transpose(-3, -2))
        resolved_logits = self.to_resolved_logits(s)

        # Weights used to compute the interface pLDDT
        ligand_weight = 2
        interface_weight = 1

        # Retrieve relevant features
        is_ligand_token = (token_type == CHAIN_TYPE_IDS["NONPOLYMER"]).float()

        # Compute the aggregated pLDDT and iPLDDT
        plddt = compute_aggregated_metric(plddt_logits)
        complex_plddt = (plddt * token_pad_mask).sum(
            dim=-1) / token_pad_mask.sum(dim=-1)

        is_contact = (d < 8).float()
        is_different_chain = (asym_id.unsqueeze(-1)
                              != asym_id.unsqueeze(-2)).float()
        is_different_chain = repeat_with_multiplicity(is_different_chain,
                                                      multiplicity)
        token_interface_mask = torch.max(
            is_contact * is_different_chain *
            (1 - is_ligand_token).unsqueeze(-1),
            dim=-1,
        ).values
        iplddt_weight = (is_ligand_token * ligand_weight +
                         token_interface_mask * interface_weight)
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(
            dim=-1) / (torch.sum(token_pad_mask * iplddt_weight, dim=-1) + 1e-5)

        # Compute the aggregated PDE and iPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = nn.functional.softmax(pred_distogram_logits,
                                                    dim=-1)
        pred_distogram_prob = repeat_with_multiplicity(pred_distogram_prob,
                                                       multiplicity)
        prob_contact = (pred_distogram_prob * self.contacts).sum(-1)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1) * token_pad_mask.unsqueeze(-2) *
            (1 - torch.eye(N_tokens, device=token_pad_mask.device)[None,
                                                                   None, :, :]))

        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(
            dim=(-2, -1)) / token_pair_mask.sum(dim=(-2, -1))
        asym_id = repeat_with_multiplicity(asym_id, multiplicity)
        token_interface_pair_mask = token_pair_mask * (asym_id.unsqueeze(-1)
                                                       != asym_id.unsqueeze(-2))
        complex_ipde = (pde * token_interface_pair_mask).sum(
            dim=(-2, -1)) / (token_interface_pair_mask.sum(dim=(-2, -1)) + 1e-5)

        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
        )
        if self.config.compute_pae:
            pae_logits = self.to_pae_logits(z)
            out_dict["pae_logits"] = pae_logits
            out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
                pae_logits, x_pred, feature_dict)
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm
        return out_dict


class Boltz1ConfidenceModule(nn.Module):

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.input_embedder_config = config.input_embedder
        self.pairformer_config = config.pairformer
        self.msa_module_config = config.msa_module
        assert self.pairformer_config.torch_dtype == self.msa_module_config.torch_dtype, f"Boltz1ConfidenceModule pairformer dtype: {self.pairformer_config.torch_dtype}, msa dtype: {self.msa_module_config.torch_dtype}"

        self.max_num_atoms_per_token = 23
        self.no_update_s = self.pairformer_config.no_update_s

        self.max_dist = self.config.max_dist
        self.num_dist_bins = self.config.num_dist_bins
        self.token_z = self.config.token_z
        self.token_s = self.config.token_s
        self.s_input_dim = (self.token_s + 2 * NUM_TOKENS + 1 +
                            NUM_POCKET_CONTACT_INFO)
        boundaries = torch.linspace(2, self.max_dist, self.num_dist_bins - 1)

        # TODO: Replace nn.Embedding with tensorrt_llm.Embedding
        self.dist_bin_pairwise_embed = nn.Embedding(self.num_dist_bins,
                                                    self.token_z)

        self.register_buffer("boundaries", boundaries)

        # Use s diffusion
        self.s_diffusion_norm = nn.LayerNorm(2 * self.token_s)
        self.s_diffusion_to_s = Linear(
            2 * self.token_s,
            self.token_s,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        self.s_to_z = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        self.s_to_z_transpose = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        self.add_s_to_z_prod = self.config.add_s_to_z_prod
        if self.add_s_to_z_prod:
            self.s_to_z_prod_in1 = Linear(
                self.s_input_dim,
                self.token_z,
                bias=False,
                dtype=self.config.torch_dtype,
                mapping=self.config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.config.skip_create_weights)
            self.s_to_z_prod_in2 = Linear(
                self.s_input_dim,
                self.token_z,
                bias=False,
                dtype=self.config.torch_dtype,
                mapping=self.config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.config.skip_create_weights)
            self.s_to_z_prod_out = Linear(
                self.token_z,
                self.token_z,
                bias=False,
                dtype=self.config.torch_dtype,
                mapping=self.config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=self.config.skip_create_weights)

        self.s_init = Linear(
            self.s_input_dim,
            self.token_s,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        self.z_init_1 = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        self.z_init_2 = Linear(
            self.s_input_dim,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        self.input_embedder = Boltz1InputEmbedder(self.input_embedder_config)
        self.rel_pos = RelativePositionEncoder(
            token_z=self.token_z,
            fix_sym_check=False,
            cyclic_pos_enc=True,
            period_broadcast=True,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            skip_create_weights=self.config.skip_create_weights)
        self.token_bonds = Linear(
            1,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        # Normalization layers
        self.s_norm = nn.LayerNorm(self.token_s,
                                   dtype=self.config.torch_dtype,
                                   eps=self.config.norm_epsilon)
        self.z_norm = nn.LayerNorm(self.token_z,
                                   dtype=self.config.torch_dtype,
                                   eps=self.config.norm_epsilon)

        # Recycling projections
        self.s_recycle = Linear(
            self.token_s,
            self.token_s,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)
        self.z_recycle = Linear(
            self.token_z,
            self.token_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        self.msa_module = MSAModule(self.msa_module_config)
        self.pairformer_module = PairformerModule(self.pairformer_config)

        self.final_s_norm = nn.LayerNorm(self.token_s,
                                         dtype=self.config.torch_dtype,
                                         eps=self.config.norm_epsilon)
        self.final_z_norm = nn.LayerNorm(self.token_z,
                                         dtype=self.config.torch_dtype,
                                         eps=self.config.norm_epsilon)

        self.confidence_heads = Boltz1ConfidenceHeads(self.config.heads)

    def load_weights(self, weights: dict) -> None:
        self.input_embedder.load_weights(weights.pop("input_embedder"))
        self.pairformer_module.load_weights(weights.pop("pairformer"))
        self.msa_module.load_weights(weights.pop("msa_module"))
        self.confidence_heads.load_weights(weights.pop("heads"))

        filter_func = lambda name, _: name.startswith(
            "msa_module") or name.startswith("pairformer") or name.startswith(
                "confidence_heads") or name.startswith("input_embedder")
        loaded_weight = recursive_calling_load_weights(self, weights,
                                                       filter_func)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def get_module_feed_dict(self, feed_dict: dict[str, torch.Tensor],
                             module_name: str) -> dict[str, Any]:
        keys = []
        if module_name == "input_embedder":
            keys = [
                "atom_to_token", "ref_pos", "atom_pad_mask", "ref_space_uid",
                "ref_charge", "ref_element", "ref_atom_name_chars", "res_type",
                "profile", "deletion_mean", "pocket_feature"
            ]
        elif module_name == "relative_position_encoding":
            keys = [
                "asym_id", "residue_index", "entity_id", "cyclic_period",
                "token_index", "sym_id"
            ]
        elif module_name == "msa_module":
            keys = [
                "msa", "has_deletion", "deletion_value", "msa_paired",
                "msa_mask"
            ]
        else:
            raise ValueError(f"Module name {module_name} not supported")
        return {key: feed_dict.get(key, None) for key in keys}

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        feature_dict: dict[str, torch.Tensor],
        pred_distogram_logits: torch.Tensor,
        multiplicity: int = 1,
        s_diffusion: Optional[torch.Tensor] = None,
        max_parallel_samples: Optional[int] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the confidence module with imitate_trunk=True.
        Args:
            s: torch.Tensor
                s from the trunk module. Shape, [B, N_tokens, token_s].
            z: torch.Tensor
                z from the trunk module. Shape, [B, N_tokens, N_tokens, token_z].
            x_pred: torch.Tensor
                x_pred from the structure module. Shape, [B, mult, N_atoms, 3] or [B*mult, N_atoms, 3].
            d: torch.Tensor
                d from the structure module.
            feature_dict: dict
                feature_dict from the dataloader.
            pred_distogram_logits: torch.Tensor
                pred_distogram_logits from the distogram module.
            multiplicity: int
                multiplicity from the structure module.
            s_diffusion: Optional[torch.Tensor]
                s_diffusion from the diffusion conditioning module.
            max_parallel_samples: Optional[int]
                max_parallel_samples from the structure module.
        return: dict[str, torch.Tensor]
            Output dictionary containing the confidence heads.
        """
        s = s.to(self.config.torch_dtype)
        z = z.to(self.config.torch_dtype)
        if max_parallel_samples is None:
            max_parallel_samples = 1
        if x_pred.ndim == 3:
            B = x_pred.shape[
                0] // multiplicity  # [B*mult, N_atoms, 3] -> [B, mult, N_atoms, 3]
            x_pred = x_pred.reshape(B, multiplicity, *x_pred.shape[1:])
        else:
            B, multiplicity, _, _ = x_pred.shape

        niter = (multiplicity + max_parallel_samples -
                 1) // max_parallel_samples
        x_chunks = x_pred.chunk(niter, dim=1)
        if s_diffusion is not None:
            if s_diffusion.ndim == 3:
                # [B, mult, ...]
                s_diffusion = s_diffusion.reshape(B, multiplicity,
                                                  *s_diffusion.shape[1:])
            s_diffusion_chunks = s_diffusion.chunk(niter, dim=1)
        else:
            s_diffusion_chunks = [None] * niter

        s_inputs = self.input_embedder(**self.get_module_feed_dict(
            feature_dict, "input_embedder"),
                                       attn_metadata=attn_metadata,
                                       all_reduce_params=all_reduce_params)
        s_init = self.s_init(s_inputs)
        z_init = (self.z_init_1(s_inputs)[:, :, None] +
                  self.z_init_2(s_inputs)[:, None, :])

        relative_position_encoding = self.rel_pos(
            **self.get_module_feed_dict(feature_dict,
                                        "relative_position_encoding"), )
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feature_dict["token_bonds"].float())

        # Apply recycling
        s = s_init + self.s_recycle(self.s_norm(s))
        z = z_init + self.z_recycle(self.z_norm(z))

        z = (z + (self.s_to_z(s_inputs)[:, :, None, :] +
                  self.s_to_z_transpose(s_inputs)[:, None, :, :]))

        if self.config.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                (self.s_to_z_prod_in1(s_inputs)[:, :, None, :] *
                 self.s_to_z_prod_in2(s_inputs)[:, None, :, :]))

        s = repeat_with_multiplicity(s, multiplicity)
        z = repeat_with_multiplicity(z, multiplicity)
        s_inputs = repeat_with_multiplicity(s_inputs, multiplicity)

        s_chunks = s.chunk(niter, dim=1)
        z_chunks = z.chunk(niter, dim=1)
        s_inputs_chunks = s_inputs.chunk(niter, dim=1)
        out_dicts = []
        for s_inputs_chunk, s_chunk, z_chunk, x_chunk, s_diffusion_chunk in zip(
                s_inputs_chunks, s_chunks, z_chunks, x_chunks,
                s_diffusion_chunks):
            n_samples = x_chunk.shape[1]

            if self.config.use_s_diffusion:
                assert s_diffusion is not None
                s_diffusion_chunk = self.s_diffusion_norm(s_diffusion_chunk)
                s_chunk = s_chunk + self.s_diffusion_to_s(s_diffusion_chunk)
            token_to_rep_atom = feature_dict["token_to_rep_atom"]
            d, distogram = compute_distogram(x_chunk,
                                             self.boundaries,
                                             token_to_rep_atom,
                                             n_samples,
                                             dtype=self.config.torch_dtype)

            distogram = self.dist_bin_pairwise_embed(distogram)
            z_chunk = z_chunk + distogram  # [B, mult, N_tokens, N_tokens, token_z]

            mask = feature_dict["token_pad_mask"]
            pair_mask = mask[:, :,
                             None] * mask[:, None, :]  # [B, N_tokens, N_tokens]
            pair_mask = repeat_with_multiplicity(
                pair_mask, n_samples)  # [B, mult, N_tokens, N_tokens]

            # Currently, MSAModule and Pairformer doesn't support multiplicity > 1, so we reshape here
            # This won't dont change the result for batching
            input_dtype = self.config.pairformer.torch_dtype
            s_chunk = s_chunk.flatten(0, 1).to(input_dtype)
            z_chunk = z_chunk.flatten(0, 1).to(input_dtype)
            s_inputs_chunk = s_inputs_chunk.flatten(0, 1).to(input_dtype)
            mask = repeat_with_multiplicity(mask,
                                            n_samples)  # [B, mult, N_tokens]
            mask = mask.flatten(0, 1).to(input_dtype)
            pair_mask = pair_mask.flatten(0, 1).to(input_dtype)

            # FIXME: create attn_metadata for msa_module and pairformer_module
            z_chunk = z_chunk + self.msa_module(
                z=z_chunk,
                emb=s_inputs_chunk,
                token_pad_mask=pair_mask,
                **self.get_module_feed_dict(feature_dict, "msa_module"),
                all_reduce_params=all_reduce_params)

            s_chunk, z_chunk = self.pairformer_module(
                s=s_chunk,
                z=z_chunk,
                mask=mask,
                pair_mask=pair_mask,
                all_reduce_params=all_reduce_params)

            # Recover dtype for the final output
            s_chunk = s_chunk.to(self.config.torch_dtype)
            z_chunk = z_chunk.to(self.config.torch_dtype)

            s_chunk, z_chunk = self.final_s_norm(s_chunk), self.final_z_norm(
                z_chunk)

            # Recover multiplicity dimension for heads
            s_chunk = s_chunk.unflatten(0, (B, n_samples))
            z_chunk = z_chunk.unflatten(0, (B, n_samples))

            out_dict = self.confidence_heads(
                s=s_chunk,
                z=z_chunk,
                x_pred=x_chunk,
                d=d,
                feature_dict=feature_dict,
                pred_distogram_logits=pred_distogram_logits,
            )
            out_dicts.append(out_dict)

        ret = concat_out_dicts(out_dicts)
        return ret
