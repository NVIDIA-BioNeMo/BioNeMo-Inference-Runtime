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

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.modules.openfold2.confidence_utils import (
    compute_plddt, compute_predicted_aligned_error, compute_tm)
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig


class AuxiliaryHeads(nn.Module):

    def __init__(self, config: BaseConfig):
        super(AuxiliaryHeads, self).__init__()

        self.config = config
        self.dtype = config.torch_dtype
        self.skip_create_weights = config.skip_create_weights
        self.epsilon = config.epsilon

        self.plddt = PerResidueLDDTCaPredictor(
            no_bins=config.per_residue_lddt.no_bins,
            c_in=config.per_residue_lddt.c_in,
            c_hidden=config.per_residue_lddt.c_hidden,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
            epsilon=self.epsilon,
        )

        self.distogram = DistogramHead(
            c_z=config.distogram.c_z,
            no_bins=config.distogram.no_bins,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        self.masked_msa = MaskedMSAHead(
            c_m=config.masked_msa.c_m,
            c_out=config.masked_msa.c_out,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            c_s=config.experimentally_resolved.c_s,
            c_out=config.experimentally_resolved.c_out,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                c_z=config.tm.c_z,
                no_bins=config.tm.no_bins,
                dtype=self.dtype,
                skip_create_weights=self.skip_create_weights,
            )

    def load_weights(self, weights: dict):

        loaded_weight = recursive_calling_load_weights(self, weights)

        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weight}")

    def forward(self, outputs: dict[str,
                                    torch.Tensor]) -> dict[str, torch.Tensor]:
        # cast the tensors to the correct dtype
        for k, v in outputs.items():
            if v.is_floating_point():
                outputs[k] = v.to(dtype=self.config.torch_dtype)
        aux_out = {}
        lddt_logits = self.plddt(outputs["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"])
        aux_out[
            "experimentally_resolved_logits"] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["ptm_score"] = compute_tm(tm_logits,
                                              no_bins=self.config.tm.no_bins)
            asym_id = outputs.get("asym_id")
            if asym_id is not None:
                aux_out["iptm_score"] = compute_tm(
                    tm_logits,
                    asym_id=asym_id,
                    interface=True,
                    no_bins=self.config.tm.no_bins)
                aux_out["weighted_ptm_score"] = (
                    self.config.tm.iptm_weight * aux_out["iptm_score"] +
                    self.config.tm.ptm_weight * aux_out["ptm_score"])

            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits, no_bins=self.config.tm.no_bins))

        return aux_out


class PerResidueLDDTCaPredictor(nn.Module):

    def __init__(
        self,
        no_bins: int,
        c_in: int,
        c_hidden: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
        epsilon: float = 1e-5,
    ):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype, eps=epsilon)

        self.linear_1 = Linear(self.c_in,
                               self.c_hidden,
                               bias=True,
                               dtype=dtype,
                               skip_create_weights=skip_create_weights)

        self.linear_2 = Linear(self.c_hidden,
                               self.c_hidden,
                               bias=True,
                               dtype=dtype,
                               skip_create_weights=skip_create_weights)
        self.linear_3 = Linear(self.c_hidden,
                               self.no_bins,
                               bias=True,
                               dtype=dtype,
                               skip_create_weights=skip_create_weights)

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(
        self,
        c_z: int,
        no_bins: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z,
                             self.no_bins,
                             bias=True,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(
        self,
        c_z: int,
        no_bins: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z,
                             self.no_bins,
                             bias=True,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        logits = self.linear(z)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(
        self,
        c_m: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super(MaskedMSAHead, self).__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m,
                             self.c_out,
                             bias=True,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s,
                             self.c_out,
                             bias=True,
                             dtype=dtype,
                             skip_create_weights=skip_create_weights)

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
