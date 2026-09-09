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
"""Protenix confidence head (AF3 Algorithm 31)."""

from typing import Any

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir.configs import BaseConfig


def _unbatch_leading(x: torch.Tensor) -> torch.Tensor:
    """Drop a leading batch dim when present (accept ``[N]`` or ``[B, N]``)."""
    return x[0] if x.dim() > 1 else x


class ProtenixConfidenceHead(nn.Module):
    """Confidence head (AF3 Algorithm 31): PAE / PDE / pLDDT / resolved.

    Head projections / LayerNorms / einsum weights run in fp32; the inner
    pairformer runs in its own (bf16) precision with ``s`` / ``z`` cast around it.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.c_s = config.c_s
        self.c_z = config.c_z
        self.c_s_inputs = config.c_s_inputs
        self.b_pae = config.b_pae
        self.b_pde = config.b_pde
        self.b_plddt = config.b_plddt
        self.b_resolved = config.b_resolved
        self.max_atoms_per_token = config.max_atoms_per_token
        skip = config.skip_create_weights

        # Distance bins (AF3 Alg. 31): [start, end) at `step`, deterministic ->
        # non-persistent buffers (not loaded from the checkpoint).
        lower_bins = torch.arange(config.distance_bin_start, config.distance_bin_end, config.distance_bin_step)
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([1e6])], dim=-1)
        self.num_bins = lower_bins.numel()
        self.register_buffer("lower_bins", lower_bins, persistent=False)
        self.register_buffer("upper_bins", upper_bins, persistent=False)

        self.linear_no_bias_s1 = Linear(
            self.c_s_inputs, self.c_z, bias=False, dtype=self.dtype, skip_create_weights=skip
        )
        self.linear_no_bias_s2 = Linear(
            self.c_s_inputs, self.c_z, bias=False, dtype=self.dtype, skip_create_weights=skip
        )
        self.linear_no_bias_d = Linear(self.num_bins, self.c_z, bias=False, dtype=self.dtype, skip_create_weights=skip)
        self.linear_no_bias_d_wo_onehot = Linear(1, self.c_z, bias=False, dtype=self.dtype, skip_create_weights=skip)

        self.pairformer_stack = PairformerModule(config.pairformer_config)
        self.pairformer_dtype = config.pairformer_config.torch_dtype

        self.linear_no_bias_pae = Linear(self.c_z, self.b_pae, bias=False, dtype=self.dtype, skip_create_weights=skip)
        self.linear_no_bias_pde = Linear(self.c_z, self.b_pde, bias=False, dtype=self.dtype, skip_create_weights=skip)
        self.plddt_weight = nn.Parameter(
            torch.empty(self.max_atoms_per_token, self.c_s, self.b_plddt, dtype=self.dtype)
        )
        self.resolved_weight = nn.Parameter(
            torch.empty(self.max_atoms_per_token, self.c_s, self.b_resolved, dtype=self.dtype)
        )

        self.input_strunk_ln = nn.LayerNorm(self.c_s, dtype=self.dtype)
        self.pae_ln = nn.LayerNorm(self.c_z, dtype=self.dtype)
        self.pde_ln = nn.LayerNorm(self.c_z, dtype=self.dtype)
        self.plddt_ln = nn.LayerNorm(self.c_s, dtype=self.dtype)
        self.resolved_ln = nn.LayerNorm(self.c_s, dtype=self.dtype)

    def _distance_embed(self, z_pair: torch.Tensor, x_rep: torch.Tensor) -> torch.Tensor:
        """Add the representative-atom distance embedding (one-hot + raw)."""
        x_rep = x_rep.to(torch.float32)
        distance = torch.cdist(x_rep, x_rep)  # [*, N_token, N_token]
        onehot = ((distance.unsqueeze(-1) > self.lower_bins) & (distance.unsqueeze(-1) < self.upper_bins)).to(
            self.dtype
        )
        z_pair = z_pair + self.linear_no_bias_d(onehot)
        z_pair = z_pair + self.linear_no_bias_d_wo_onehot(distance.unsqueeze(-1).to(self.dtype))
        return z_pair

    def _per_sample(
        self,
        atom_to_token_idx: torch.Tensor,
        atom_to_tokatom_idx: torch.Tensor,
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        x_rep: torch.Tensor,
    ) -> tuple:
        """AF3 Alg. 31 for one diffusion sample (memory-efficient path)."""
        z_pair = self._distance_embed(z_pair, x_rep)

        s_single, z_pair = self.pairformer_stack(
            s_trunk.to(self.pairformer_dtype), z_pair.to(self.pairformer_dtype), single_mask, pair_mask
        )
        z_pair = z_pair.to(self.dtype)
        s_single = s_single.to(self.dtype)

        pae = self.linear_no_bias_pae(self.pae_ln(z_pair))
        pde = self.linear_no_bias_pde(self.pde_ln(z_pair + z_pair.transpose(-2, -3)))
        a = s_single[..., atom_to_token_idx, :]  # [*, N_atom, c_s]
        plddt = torch.einsum("...nc,ncb->...nb", self.plddt_ln(a), self.plddt_weight[atom_to_tokatom_idx])
        resolved = torch.einsum("...nc,ncb->...nb", self.resolved_ln(a), self.resolved_weight[atom_to_tokatom_idx])
        return plddt, pae, pde, resolved

    def prepare(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        single_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Sample-independent inputs shared across diffusion samples.

        Args:
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]``

        Returns:
            Context dict with trunk ``s_trunk`` / pair ``z``
            ``[B, N_token, N_token, c_z]``, masks, and unbatched atom index maps.
        """
        s_inputs = s_inputs.to(self.dtype)
        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk.to(self.dtype), min=-512, max=512))

        z = self.linear_no_bias_s1(s_inputs).unsqueeze(-3) + self.linear_no_bias_s2(s_inputs).unsqueeze(-2)
        z = z + z_trunk.to(self.dtype)

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])
        if single_mask is None:
            single_mask = s_trunk.new_ones(s_trunk.shape[:-1])

        return {
            "s_trunk": s_trunk,
            "z": z,
            "single_mask": single_mask,
            "pair_mask": pair_mask,
            "rep_mask": _unbatch_leading(input_feature_dict["distogram_rep_atom_mask"]).bool(),
            "atom_to_token_idx": _unbatch_leading(input_feature_dict["atom_to_token_idx"]),
            "atom_to_tokatom_idx": _unbatch_leading(input_feature_dict["atom_to_tokatom_idx"]),
        }

    def per_sample_logits(self, ctx: dict[str, Any], x_pred_coords_i: torch.Tensor) -> tuple:
        """PAE / PDE / pLDDT / resolved logits for one diffusion sample.

        Args:
            ctx: :meth:`prepare` context
            x_pred_coords_i: ``[B, N_atom, 3]`` (one sample)

        Returns:
            ``plddt`` ``[B, N_atom, b_plddt]``,
            ``pae`` ``[B, N_token, N_token, b_pae]``,
            ``pde`` ``[B, N_token, N_token, b_pde]``,
            ``resolved`` ``[B, N_atom, b_resolved]``
        """
        x_rep = x_pred_coords_i[..., ctx["rep_mask"], :]
        return self._per_sample(
            ctx["atom_to_token_idx"],
            ctx["atom_to_tokatom_idx"],
            ctx["s_trunk"],
            ctx["z"],
            ctx["single_mask"],
            ctx["pair_mask"],
            x_rep,
        )

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        x_pred_coords: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        single_mask: torch.Tensor | None = None,
    ) -> dict:
        """Stacked entry point (returns ``[B, N_sample, ...]`` logits).

        Args:
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]``
            x_pred_coords: ``[B, N_sample, N_atom, 3]``

        Returns:
            ``plddt_logits`` ``[B, N_sample, N_atom, b_plddt]``,
            ``pae_logits`` / ``pde_logits``
            ``[B, N_sample, N_token, N_token, b_*]``,
            ``resolved_logits`` ``[B, N_sample, N_atom, b_resolved]``
        """
        ctx = self.prepare(input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, single_mask)
        n_sample = x_pred_coords.size(-3)

        plddt, pae, pde, resolved = [], [], [], []
        for i in range(n_sample):
            p, a, d, r = self.per_sample_logits(ctx, x_pred_coords[..., i, :, :])
            plddt.append(p)
            pae.append(a)
            pde.append(d)
            resolved.append(r)
        return {
            "plddt_logits": torch.stack(plddt, dim=-3),
            "pae_logits": torch.stack(pae, dim=-4),
            "pde_logits": torch.stack(pde, dim=-4),
            "resolved_logits": torch.stack(resolved, dim=-3),
        }
