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
"""Protenix recycling trunk (template embedder + MSA module + pairformer)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import PrecomputedPairMasks, precompute_pair_masks
from bionemo_ir._torch.graph_optimization.region import GraphRegion
from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.token_padding import pad_trunk_tokens, unpad_trunk_tokens
from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir._torch.layers.triangle_nodes import set_trimul_token_padding
from bionemo_ir._torch.modules.openfold3.trunk import MSAModuleStack
from bionemo_ir._torch.modules.protenix.template import ProtenixTemplateEmbedder
from bionemo_ir.configs import BaseConfig
from bionemo_ir.utils import str_dtype_to_torch


class ProtenixMSAModule(nn.Module):
    """Protenix MSA module (AF3 Algorithm 8).

    Feature embedding (one-hot MSA + deletion -> ``linear_no_bias_m``, plus
    ``linear_no_bias_s``) then shared :class:`MSAModuleStack`. MSA row
    subsampling is handled upstream by the data pipeline.
    """

    # OSS Algorithm 8 raw MSA feature widths (concat order feeds linear_no_bias_m).
    input_feature: dict[str, int] = {
        "msa": 32,
        "has_deletion": 1,
        "deletion_value": 1,
    }

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.n_blocks = config.no_blocks
        self.c_m = config.c_m
        self.dtype = config.torch_dtype

        self.linear_no_bias_m = Linear(
            config.msa_input_dim, self.c_m, bias=False, dtype=self.dtype, skip_create_weights=config.skip_create_weights
        )
        self.linear_no_bias_s = Linear(
            config.c_s_inputs, self.c_m, bias=False, dtype=self.dtype, skip_create_weights=config.skip_create_weights
        )
        self.msa_stack = MSAModuleStack(config)

    def _embed_msa(self, input_feature_dict: dict[str, Any], s_inputs: torch.Tensor) -> torch.Tensor:
        """Build the MSA embedding ``m`` ``[B, N_msa, N_token, c_m]``."""
        msa = input_feature_dict["msa"].long()
        target_shape = msa.shape
        linear = self.linear_no_bias_m
        weight = linear.weight

        # one_hot(msa, 32) @ W == embedding(msa, W.T). Accumulate deletion
        # columns directly — avoids int64 one-hot and concat bf16 [B,S,N,34].
        symbol_cols = self.input_feature["msa"]
        m = F.embedding(msa, weight[:, :symbol_cols].t())
        for offset, name in enumerate(("has_deletion", "deletion_value")):
            feature = input_feature_dict[name].reshape(*target_shape, self.input_feature[name]).to(weight.dtype)
            m.addcmul_(feature, weight[:, symbol_cols + offset])

        m.add_(self.linear_no_bias_s(s_inputs.to(self.dtype)).unsqueeze(1))
        return m

    def build_pair_masks(self, pair_mask: torch.Tensor) -> PrecomputedPairMasks | None:
        """Precompute pair-stack triangle-attention mask bias (reuse every cycle)."""
        if self.n_blocks < 1:
            return None
        block = self.msa_stack.blocks[0]
        return precompute_pair_masks(
            block.triangle_attn_backend, pair_mask.to(self.dtype), inf=block.inf, dtype=block.dtype
        )

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        msa_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
    ) -> torch.Tensor:
        """Update pair ``z`` via MSA feature embed + :class:`MSAModuleStack`.

        Args:
            input_feature_dict: ``msa`` ``[B, N_msa, N_token]``,
                ``has_deletion`` / ``deletion_value`` ``[B, N_msa, N_token]``
            z: ``[B, N_token, N_token, c_z]``
            s_inputs: ``[B, N_token, c_s_inputs]``

        Returns:
            ``[B, N_token, N_token, c_z]`` updated pair (unchanged if no MSA)
        """
        if self.n_blocks < 1 or "msa" not in input_feature_dict:
            return z

        m = self._embed_msa(input_feature_dict, s_inputs)
        if msa_mask is None:
            msa_mask = m.new_ones(m.shape[:-1])
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])
        out = self.msa_stack(m, z, msa_mask, pair_mask, attn_metadata, precomputed_masks=precomputed_masks)
        return out.to(z.dtype)


class ProtenixTrunk(nn.Module):
    """Protenix recycling trunk (OSS ``Protenix`` Lines 7-13).

    Each cycle: re-project recycled ``z``, optional template, MSA, project
    recycled ``s``, pairformer. Owns template embedder, MSA module, pairformer.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.n_cycle = config.n_cycle
        # Gate template in the recycling loop (OSS default inference uses
        # use_template=false). Module + weights still built/loaded.
        self.use_template = getattr(config, "use_template", False)
        c_s, c_z = config.c_s, config.c_z
        dtype = config.torch_dtype
        self.dtype = dtype
        self.pair_state_dtype = str_dtype_to_torch(config.pair_state_dtype)

        self.template_embedder = ProtenixTemplateEmbedder(config.template_embedder_config)
        self.msa_module = ProtenixMSAModule(config.msa_module_config)
        self.pairformer_stack = PairformerModule(config.pairformer_config)
        self.pairformer_dtype = config.pairformer_config.torch_dtype
        self.enable_token_pad = config.pairformer_config.enable_token_pad
        self.token_pad_spec = config.token_pad_spec

        self.layernorm_z_cycle = nn.LayerNorm(c_z, eps=config.norm_epsilon, dtype=dtype)
        self.linear_no_bias_z_cycle = Linear(
            c_z, c_z, bias=False, dtype=dtype, skip_create_weights=config.skip_create_weights
        )
        self.layernorm_s = nn.LayerNorm(c_s, eps=config.norm_epsilon, dtype=dtype)
        self.linear_no_bias_s = Linear(
            c_s, c_s, bias=False, dtype=dtype, skip_create_weights=config.skip_create_weights
        )
        # forward() pads the tokens before any stack runs.
        set_trimul_token_padding(self, self.enable_token_pad)
        self.graph = GraphRegion(self, "_recycle_step", config.graph_optimization_config)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        num_cycles: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``n_cycle`` recycling iterations; return trunk ``(s, z)``.

        Args:
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_init: ``[B, N_token, c_s]``
            z_init: ``[B, N_token, N_token, c_z]``
            num_cycles: override ``config.n_cycle`` when set

        Returns:
            ``s`` ``[B, N_token, c_s]``, ``z`` ``[B, N_token, N_token, c_z]``
        """
        # Persistent pair state in configured storage dtype (fp32 default;
        # bf16 opt-in). Recycling/template projections stay in self.dtype.
        z_init = z_init.to(self.pair_state_dtype)
        if pair_mask is None:
            pair_mask = z_init.new_ones(z_init.shape[:-1])
        if token_mask is None:
            token_mask = s_init.new_ones(s_init.shape[:-1])

        n_true = z_init.shape[1]
        if self.enable_token_pad:
            padded, input_feature_dict, n_true = pad_trunk_tokens(
                {
                    "s_inputs": s_inputs,
                    "s_init": s_init,
                    "z_init": z_init,
                    "pair_mask": pair_mask,
                    "token_mask": token_mask,
                },
                n_true,
                self.token_pad_spec,
                input_feature_dict,
            )
            s_inputs, s_init, z_init, pair_mask, token_mask = (
                padded["s_inputs"],
                padded["s_init"],
                padded["z_init"],
                padded["pair_mask"],
                padded["token_mask"],
            )
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        msa_precomputed = self.msa_module.build_pair_masks(pair_mask)
        use_template = (
            self.use_template and self.template_embedder.n_blocks > 0 and "template_aatype" in input_feature_dict
        )
        # Template features do not change across cycles, so find the duplicates once.
        template_representatives = (
            self.template_embedder.template_representatives(input_feature_dict) if use_template else None
        )

        msa_mask_tensors = (
            (msa_precomputed.pair_mask, msa_precomputed.mask_bias, msa_precomputed.mask_bias_transposed)
            if msa_precomputed is not None
            else None
        )
        cycle_features = {
            name: value
            for name, value in input_feature_dict.items()
            if name in ("msa", "has_deletion", "deletion_value", "asym_id") or name.startswith("template_")
        }

        n_cycle = self.n_cycle if num_cycles is None else num_cycles
        for _ in range(n_cycle):
            s, z = self.graph(
                s=s,
                z=z,
                s_init=s_init,
                z_init=z_init,
                s_inputs=s_inputs,
                input_feature_dict=cycle_features,
                pair_mask=pair_mask,
                token_mask=token_mask,
                attn_metadata=attn_metadata,
                msa_mask_tensors=msa_mask_tensors,
                template_representatives=template_representatives,
            )
        s, z = unpad_trunk_tokens(s, z, n_true=n_true)
        return s, z

    def _recycle_step(
        self,
        *,
        s: torch.Tensor,
        z: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s_inputs: torch.Tensor,
        input_feature_dict: dict[str, torch.Tensor],
        pair_mask: torch.Tensor,
        token_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        msa_mask_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        template_representatives: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a recycle using tensor-only precomputed masks that replay refreshes."""
        precomputed = PrecomputedPairMasks(*msa_mask_tensors) if msa_mask_tensors is not None else None
        z = self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z.to(self.dtype)))
        z.add_(z_init)
        if template_representatives is not None:
            z.add_(self.template_embedder(input_feature_dict, z, pair_mask, template_representatives))
        z = self.msa_module(
            input_feature_dict,
            z.to(self.pair_state_dtype),
            s_inputs,
            pair_mask=pair_mask,
            attn_metadata=attn_metadata,
            precomputed_masks=precomputed,
        )
        s = self.linear_no_bias_s(self.layernorm_s(s))
        s.add_(s_init)
        s_pf, z_pf = self.pairformer_stack(
            s.to(self.pairformer_dtype), z.to(self.pairformer_dtype), token_mask, pair_mask
        )
        return s_pf.to(self.dtype), z_pf.to(self.pair_state_dtype)
