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

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerModule
from tensorrt_bionemo._torch.modules.openfold3.utils.atomize_utils import (
    broadcast_token_feat_to_atoms, get_token_representative_atoms,
    max_atom_per_token_masked_select)
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.models.openfold3.config import PairformerConfig


class PairformerEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 31, line 1 - 6
    """

    def __init__(self,
                 pairformer: PairformerConfig,
                 c_s_input: int,
                 c_z: int,
                 min_bin: float,
                 max_bin: float,
                 no_bin: int,
                 inf: float,
                 dtype: torch.dtype = torch.float32,
                 skip_create_weights: bool = False):
        """
        Args:
            pairformer:
                Config for PairFormerStack used
            c_s_input:
                Single (input) embedding dimension
            c_z:
                Pair embedding dimension
            min_bin:
                Minimum value for bin (3.25). The value is slightly
                different from SI. Previous AF2 implementation utilized these values
                for bins.
            max_bin:
                Maximum value for bin (20.75). ibid
            no_bin:
                Number of bins (15). ibid
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bin = no_bin
        self.inf = inf
        self.dtype = dtype
        self.skip_create_weights = skip_create_weights

        self.linear_i = Linear(c_s_input,
                               c_z,
                               bias=False,
                               dtype=self.dtype,
                               skip_create_weights=self.skip_create_weights)

        self.linear_j = Linear(c_s_input,
                               c_z,
                               bias=False,
                               dtype=self.dtype,
                               skip_create_weights=self.skip_create_weights)

        self.linear_distance = Linear(
            self.no_bin,
            c_z,
            bias=False,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights)

        bins = torch.linspace(min_bin, max_bin, no_bin)
        squared_bins = bins**2
        upper = torch.cat([squared_bins[1:],
                           squared_bins.new_tensor([inf])],
                          dim=-1)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("squared_bins", squared_bins, persistent=False)
        self.register_buffer("upper", upper, persistent=False)
        self.pairformer_stack = PairformerModule(config=pairformer)

    def embed_zij(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
    ):
        orig_dtype = zij.dtype
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            # si projection to zij
            zij = (zij + self.linear_i(si_input.unsqueeze(-2)) +
                   self.linear_j(si_input.unsqueeze(-3)))

            # Embed pair distances of representative atoms
            dij = torch.sum(
                (x_pred[..., None, :] - x_pred[..., None, :, :])**2,
                dim=-1,
                keepdims=True,
            )
            dij = ((dij > self.squared_bins) * (dij < self.upper)).type(
                x_pred.dtype)
            zij = zij + self.linear_distance(dij)

        return zij.to(dtype=orig_dtype)

    def per_sample_pairformer_emb(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ):
        """Memory-efficient path: run pairformer per diffusion sample.

        Instead of expanding zij across all samples (O(samples * N^2 * C_z)),
        processes one sample at a time to keep peak memory at O(N^2 * C_z).

        x_pred is [B, num_samples, N_token, 3].  si/zij/single_mask may
        carry a leading sample dim of size 1 (from unsqueeze in the model
        forward) or the full sample dim.  We strip it before the loop so
        the pairformer always sees plain [B, N, N, C_z] / [B, N, C_s].
        """
        no_samples = x_pred.shape[-3]

        # The TRT-BNM model unsqueezes a sample dim (size 1) onto trunk
        # outputs and the batch before calling the confidence heads.
        # Strip that dim so the pairformer sees plain [B, N, ...] tensors.
        def _strip_sample_dim(t: torch.Tensor, expected_ndim: int):
            while t.ndim > expected_ndim and t.shape[-(expected_ndim +
                                                       1)] == 1:
                t = t.squeeze(-(expected_ndim + 1))
            return t

        si_input = _strip_sample_dim(si_input, 3)  # -> [B, N, C_s]
        si = _strip_sample_dim(si, 3)  # -> [B, N, C_s]
        zij = _strip_sample_dim(zij, 4)  # -> [B, N, N, C_z]
        pair_mask = _strip_sample_dim(pair_mask, 3)  # -> [B, N, N]
        # single_mask may be [B, num_samples, N] (from repr atoms); take first sample
        if single_mask.ndim > 2:
            single_mask = single_mask[:, 0]

        si_list = []
        zij_list = []

        for i in range(no_samples):
            zij_chunk = self.embed_zij(
                si_input=si_input,
                zij=zij,
                x_pred=x_pred[:, i:i + 1],
            )
            # embed_zij broadcasts to [B, 1, N, N, C_z]; squeeze sample dim
            zij_chunk = zij_chunk.squeeze(-4)
            si_chunk = si

            si_chunk, zij_chunk = self.pairformer_stack(
                si_chunk,
                zij_chunk,
                single_mask,
                pair_mask,
            )

            si_list.append(si_chunk.unsqueeze(-3))
            zij_list.append(zij_chunk.unsqueeze(-4))

            del si_chunk, zij_chunk

        # [B, num_samples, N, C_s] and [B, num_samples, N, N, C_z]
        return torch.cat(si_list, dim=-3), torch.cat(zij_list, dim=-4)

    def pairformer_emb(self, si_input: torch.Tensor, si: torch.Tensor,
                       zij: torch.Tensor, x_pred: torch.Tensor,
                       single_mask: torch.Tensor, pair_mask: torch.Tensor):
        zij = self.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)
        batch_dims = x_pred.shape[:-2]

        def reshape_inputs(x: torch.Tensor, feat_dims: list):
            x = x.expand(*(batch_dims + feat_dims))
            x = x.reshape(-1, *feat_dims)
            return x

        def reshape_outputs(x: torch.Tensor, feat_dims: list):
            return x.reshape(*batch_dims, *feat_dims)

        si = reshape_inputs(x=si, feat_dims=si.shape[-2:])
        zij = reshape_inputs(x=zij, feat_dims=zij.shape[-3:])
        single_mask = reshape_inputs(x=single_mask,
                                     feat_dims=single_mask.shape[-1:])
        pair_mask = reshape_inputs(x=pair_mask, feat_dims=pair_mask.shape[-2:])
        si, zij = self.pairformer_stack(si, zij, single_mask, pair_mask)

        si = reshape_outputs(x=si, feat_dims=si.shape[-2:])
        zij = reshape_outputs(x=zij, feat_dims=zij.shape[-3:])

        return si, zij

    def forward(self,
                si_input: torch.Tensor,
                si: torch.Tensor,
                zij: torch.Tensor,
                x_pred: torch.Tensor,
                single_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                apply_per_sample: bool = False):
        """
        Args:
            si_input:
                [*, N_token, C_s] Output of InputFeatureEmbedder
            si:
                [*, N_token, C_s] Single embedding
            zij:
                [*, N_token, N_token, C_z] Pairwise embedding
            x_pred:
                Representative atom predicted coordinates per token.
                Shape: [*, num_samples, N_token, 3] when apply_per_sample=True,
                or [*, N_token, 3] when apply_per_sample=False (expanded internally).
            single_mask:
                [*, N_token] Single mask
            pair_mask:
                [*, N_token, N_token] Pair mask
            apply_per_sample:
                When True, run pairformer embedding per diffusion sample
                to avoid OOM from expanding zij across all samples.

        Returns:
            si:
                [*, N_token, C_s] Updated single representation
            zij:
                [*, N_token, N_token, C_z] Updated pair representation
        """
        if apply_per_sample:
            si, zij = self.per_sample_pairformer_emb(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=x_pred,
                single_mask=single_mask,
                pair_mask=pair_mask,
            )
        else:
            si, zij = self.pairformer_emb(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=x_pred,
                single_mask=single_mask,
                pair_mask=pair_mask,
            )

        return si, zij


class PredictedAlignedErrorHead(nn.Module):
    """
    Implements PredictedAlignedError Head (Algorithm 31, Line 5) for
    AF3 (subsection 4.3.2)
    """

    def __init__(self,
                 c_z: int,
                 c_out: int,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-5,
                 skip_create_weights: bool = False):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedAlignedError (PAE) bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_z, dtype=dtype, eps=eps)
        self.linear = Linear(self.c_z,
                             self.c_out,
                             bias=False,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def _compute_logits(self, zij: torch.Tensor):
        logits = self.linear(self.layer_norm(zij))
        return logits

    def forward(self, zij):
        """
        Args:
            zij:
                [*, N, N, C_z] Pair embedding
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """

        logits = self._compute_logits(zij=zij)

        return logits


class PredictedDistanceErrorHead(nn.Module):
    """
    Implements PredictedDistanceError Head (Algorithm 31, Line 6) for
    AF3 (subsection 4.3.3)
    """

    def __init__(self,
                 c_z: int,
                 c_out: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = torch.float32,
                 skip_create_weights: bool = False):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedDistanceError (PDE) bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_z, dtype=dtype, eps=eps)
        self.linear = Linear(self.c_z,
                             self.c_out,
                             bias=False,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def _compute_logits(self, zij: torch.Tensor):
        logits = self.linear(self.layer_norm(zij))
        logits = logits + logits.transpose(-2, -3)
        return logits

    def forward(self, zij):
        """
        Args:
            zij:
                [*, N, N, C_z] Pair embedding
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """

        logits = self._compute_logits(zij=zij)

        return logits


class PerResidueLDDTAllAtom(nn.Module):
    """
    Implements Plddt Head (Algorithm 31, Line 7) for AF3 (subsection 4.3.1)
    """

    def __init__(self,
                 c_s: int,
                 c_out: int,
                 max_atoms_per_token: int,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-5,
                 skip_create_weights: bool = False):
        """
        Args:
            c_s:
                Input channel dimension
            max_atoms_per_token:
                Maximum atoms per token
            c_out:
                Number of PLDDT bins
        """
        super().__init__()

        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_s, dtype=dtype, eps=eps)
        self.linear = Linear(self.c_s,
                             self.max_atoms_per_token * self.c_out,
                             bias=False,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, s: torch.Tensor, max_atom_per_token_mask: torch.Tensor):
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            max_atom_per_token_mask:
                [*, N_token * max_atoms_per_token] Flat mask of atoms per token
                padded to max_atoms_per_token
        Returns:
            logits:
                [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # Flatten batch dims
        max_atom_per_token_mask = max_atom_per_token_mask.reshape(
            -1, n_token * self.max_atoms_per_token)

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(*batch_dims,
                                n_token * self.max_atoms_per_token, self.c_out)

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
        )

        return logits


class ExperimentallyResolvedHeadAllAtom(nn.Module):
    """
    Implements resolvedHeads for AF3, subsection 4.3.3
    """

    def __init__(self,
                 c_s: int,
                 c_out: int,
                 max_atoms_per_token: int,
                 dtype: torch.dtype = torch.float32,
                 eps: float = 1e-5,
                 skip_create_weights: bool = False):
        """
        Args:
            c_s:
                Input channel dimension
            max_atoms_per_token:
                Maximum atoms per token
            c_out:
                Number of ExperimentallyResolved Head AllAtom bins
        """
        super().__init__()

        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_s, dtype=dtype, eps=eps)
        self.linear = Linear(self.c_s,
                             self.max_atoms_per_token * self.c_out,
                             bias=False,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, s: torch.Tensor, max_atom_per_token_mask: torch.Tensor):
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            max_atom_per_token_mask:
                [*, N_token * max_atoms_per_token] Flat mask of atoms per token
                padded to max_atoms_per_token
        Returns:
            logits:
                [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # Flatten batch dims
        max_atom_per_token_mask = max_atom_per_token_mask.reshape(
            -1, n_token * self.max_atoms_per_token)

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(*batch_dims,
                                n_token * self.max_atoms_per_token, self.c_out)

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
        )

        return logits


class DistogramHead(nn.Module):
    """
    Implementation of distogram head for both AF2 and AF3.

    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2), section 4.4 (AF3)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z,
                             self.c_out,
                             bias=False,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, z):
        """
        Args:
            z:
                [*, N, N, C_z] Pair embedding
        Returns:
            logit:
                [*, N, N, C_out] Distogram probability distribution

        Note:
            For symmetric pairwise PairDistanceError loss (PDE),
            logits are calculated by linear(zij + zij.transpose(-2, -3))
            In SI this happens before the linear layer is applied.
        """

        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class AuxiliaryHeadsAllAtom(nn.Module):
    """
    Auxiliary head for OF3
    Implements AF3 Algorithm 31 with main inference loop (Algorithm 1) line 16 - 17.
    """

    def __init__(self, config):
        """
        Args:
            config: ConfigDict with following keys
                "pairformer_embedding": Pairformer embedding config
                "pae": PAE config
                "pde": PDE config
                "lddt": LDDT config
                "distogram": Distogram config
                "experimentally_resolved": Experimentally_resolved config
        """
        super().__init__()
        self.config = config
        self.max_atoms_per_token = config.max_atoms_per_token
        self.dtype = config.torch_dtype
        self.skip_create_weights = config.skip_create_weights
        # memory_efficient_mode default to True. This mean we will run pairformer_embedding with sequential mode (for each diffusion sample)
        self.apply_per_sample = config.memory_efficient_mode

        self.pairformer_embedding = PairformerEmbedding(
            pairformer=config.pairformer,
            c_s_input=config.c_s_input,
            c_z=config.c_z,
            min_bin=config.min_bin,
            max_bin=config.max_bin,
            no_bin=config.no_bin,
            inf=config.inf,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights)

        self.pde = PredictedDistanceErrorHead(
            c_z=config.pde.c_z,
            c_out=config.pde.c_out,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights)

        self.plddt = PerResidueLDDTAllAtom(
            c_s=config.lddt.c_s,
            c_out=config.lddt.c_out,
            max_atoms_per_token=config.lddt.max_atoms_per_token,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights)

        self.distogram = DistogramHead(
            c_z=config.distogram.c_z,
            c_out=config.distogram.c_out,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights)

        self.experimentally_resolved = ExperimentallyResolvedHeadAllAtom(
            c_s=config.experimentally_resolved.c_s,
            c_out=config.experimentally_resolved.c_out,
            max_atoms_per_token=config.experimentally_resolved.
            max_atoms_per_token,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights)

        if config.pae.enabled:
            self.pae = PredictedAlignedErrorHead(
                c_z=config.pae.c_z,
                c_out=config.pae.c_out,
                dtype=self.dtype,
                skip_create_weights=self.skip_create_weights)

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(self, batch: dict, si_input: torch.Tensor, output: dict):
        """
        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            output:
                Dict containing outputs
                    "si_trunk" ([*, N_token, C_s]):
                        Single representation output from model trunk
                    "zij_trunk" ([*, N_token, N_token, C_z]):
                        Pair representation output from model trunk
                    "atom_positions_predicted" ([*, N_atom, 3]):
                        Predicted atom positions

        Returns:
            aux_out:
                Dict containing following keys:
                    "plddt_logits" ([*, N_atom, 50]):
                        Predicted binned PLDDT logits
                    "pae_logits" ([*, N_token, N_token, 64]):
                        Predicted binned PAE logits
                    "pde_logits" ([*, N_token, N_token, 64]):
                        Predicted binned PDE logits
                    "experimentally_resolved_logits" ([*, N_atom, 2]):
                        Predicted binned experimentally resolved logits
                    "distogram_logits" ([*, N_token, N_token, 64]):
                        Predicted binned distogram logits
        Note:
            Previous implementations of losses include softmax so all
            heads return logits.
        """
        aux_out = {}

        out_dtype = output["atom_positions_predicted"].dtype
        si = output["si_trunk"].to(dtype=self.dtype)
        zij = output["zij_trunk"].to(dtype=self.dtype)
        atom_positions_predicted = output["atom_positions_predicted"].to(
            dtype=si.dtype)

        # Distogram head: Main loop (Algorithm 1), line 17
        distogram_logits = self.distogram(z=zij)

        aux_out["distogram_logits"] = distogram_logits

        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Get representative atoms
        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch,
            x=atom_positions_predicted,
            atom_mask=batch["atom_mask"])

        out_device = atom_positions_predicted.device

        # Embed trunk outputs
        si, zij = self.pairformer_embedding(
            si_input=si_input.to(dtype=self.dtype),
            si=si.to(dtype=self.dtype),
            zij=zij.to(dtype=self.dtype),
            x_pred=repr_x_pred.to(dtype=self.dtype),
            single_mask=repr_x_mask.to(dtype=self.dtype),
            pair_mask=pair_mask.to(dtype=self.dtype),
            apply_per_sample=self.apply_per_sample,
        )

        # Get atom mask padded to MAX_ATOMS_PER_TOKEN
        # Required to extract pLDDT and experimentally resolved logits for
        # the flat atom representation

        max_atom_per_token_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=token_mask,
            max_num_atoms_per_token=self.max_atoms_per_token,
        )

        si = si.to(device=out_device)
        aux_out["plddt_logits"] = self.plddt(
            s=si, max_atom_per_token_mask=max_atom_per_token_mask)

        experimentally_resolved_logits = self.experimentally_resolved(
            si, max_atom_per_token_mask)
        aux_out[
            "experimentally_resolved_logits"] = experimentally_resolved_logits

        pde_logits = self.pde(zij)

        if self.config.pae.enabled:
            aux_out["pae_logits"] = self.pae(zij).to(device=out_device)

        aux_out["pde_logits"] = pde_logits.to(device=out_device)

        aux_out = {k: v.to(dtype=out_dtype) for k, v in aux_out.items()}

        return aux_out
