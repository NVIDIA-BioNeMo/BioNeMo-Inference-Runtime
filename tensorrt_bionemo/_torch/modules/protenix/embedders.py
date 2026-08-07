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

"""Protenix input embedding modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.modules.protenix.atom_attention import ProtenixAtomAttentionEncoder
from tensorrt_bionemo.configs import BaseConfig


class ProtenixInputFeatureEmbedder(nn.Module):
    """Protenix Algorithm 2 input feature embedder.

    Returns only ``s_inputs`` (``c_token + 32 + 32 + 1``). Pair init / RPE /
    token-bond live at the parent model (unlike OF3 ``InputEmbedderAllAtom``).
    """

    input_feature_dims: dict[str, int] = {
        "restype": 32,
        "profile": 32,
        "deletion_mean": 1,
    }

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.c_token = config.c_token
        self.esm_enabled = config.esm_enabled
        self.esm_embedding_dim = config.esm_embedding_dim
        self.atom_attention_encoder = ProtenixAtomAttentionEncoder(config)
        if self.esm_enabled:
            self.linear_esm = nn.Linear(
                self.esm_embedding_dim,
                self.c_token + sum(self.input_feature_dims.values()),
                bias=False,
                dtype=config.torch_dtype,
            )
            nn.init.zeros_(self.linear_esm.weight)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Embed Protenix token input features → ``s_inputs``.

        Args:
            input_feature_dict: atom ref features + ``restype`` / ``profile`` /
                ``deletion_mean`` ``[..., N_token, *]`` (optional ESM)

        Returns:
            ``[..., N_token, c_token + 65]`` (``c_token + restype32 + profile32
            + deletion_mean1``; equals ``c_s_inputs`` when ESM is off)
        """
        atom_token_embedding, _, _, _ = self.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"],
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
            attn_metadata=attn_metadata,
        )

        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [atom_token_embedding]
            + [input_feature_dict[name].reshape(*batch_shape, dim) for name, dim in self.input_feature_dims.items()],
            dim=-1,
        )

        if self.esm_enabled:
            s_inputs = s_inputs + self.linear_esm(input_feature_dict["esm_token_embedding"])

        return s_inputs


class ProtenixConstraintEmbedder(nn.Module):
    """Optional constraint pair embedders (pocket / contact / contact-atom).

    protenix-v2 disables all of them (no parameters; ``forward`` returns ``None``).
    Substructure embedder is not ported — enabling it raises.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        c = config.c_constraint_z
        dtype = config.torch_dtype
        self.pocket_enable = config.pocket_enable
        self.contact_enable = config.contact_enable
        self.contact_atom_enable = config.contact_atom_enable
        if config.substructure_enable:
            raise NotImplementedError(
                "Protenix substructure constraint embedder is not ported (disabled in protenix-v2)."
            )

        def _embedder(c_in: int):
            return Linear(c_in, c, bias=False, dtype=dtype, skip_create_weights=config.skip_create_weights)

        if self.pocket_enable:
            self.pocket_z_embedder = _embedder(config.pocket_c_z_input)
        if self.contact_enable:
            self.contact_z_embedder = _embedder(config.contact_c_z_input)
        if self.contact_atom_enable:
            self.contact_atom_z_embedder = _embedder(config.contact_atom_c_z_input)

    def forward(self, constraint_feature_dict: dict[str, Any]) -> torch.Tensor | None:
        """Sum enabled constraint pair projections, or ``None`` if none enabled.

        Args:
            constraint_feature_dict: optional ``pocket`` / ``contact`` /
                ``contact_atom`` features

        Returns:
            ``z_constraint`` ``[..., N_token, N_token, c_constraint_z]``, or
            ``None`` when all constraint embedders are disabled
        """
        z_constraint: torch.Tensor | None = None

        def _add(z: torch.Tensor | None, update: torch.Tensor) -> torch.Tensor:
            return update if z is None else z + update

        if self.pocket_enable:
            z_constraint = _add(z_constraint, self.pocket_z_embedder(constraint_feature_dict["pocket"]))
        if self.contact_enable:
            z_constraint = _add(z_constraint, self.contact_z_embedder(constraint_feature_dict["contact"]))
        if self.contact_atom_enable:
            z_constraint = _add(z_constraint, self.contact_atom_z_embedder(constraint_feature_dict["contact_atom"]))
        return z_constraint
