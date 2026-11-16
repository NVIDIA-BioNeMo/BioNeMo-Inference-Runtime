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
"""Reference distribution for RFProteína's product space.

Flow matching integrates from a reference distribution at ``t = 0`` toward the
data distribution at ``t = 1``. This module owns the ``t = 0`` end: it draws the
product-space reference noise, and resamples a backbone from its forward marginal
for inverse folding. A host that starts a rollout from noise can call
:meth:`~RFProteinaProductSpaceFlowMatcher.sample_noise`; the model's simulator takes
its start state from the caller. The integration itself belongs to the shared
sampling runtime, so nothing here iterates or touches a schedule.

The draw is more than a Gaussian because RFProteína centers it. The backbone is
projected onto the zero-center-of-mass subspace, since a translated backbone is
the same structure and the network should never have to learn that. On top of
that, *stochastic centering* offsets the generated tokens by a small random
translation, which is what teaches the model that absolute position carries no
meaning; the scope decides whether one offset covers everything, whether
coordinate-conditioned tokens are pinned at the origin, or whether each chain or
coordinate group moves independently.

Draw order is part of the contract. Modalities are drawn in
:data:`PRODUCT_SPACE_MODALITIES` order and each modality consumes its noise
before the next begins, so a given seed reproduces the same reference state as
upstream's ``ProductSpaceFlowMatcher.sample_noise``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.sampling.random import randn

from .contracts import ModalityReference, RFProteinaState
from .groups import compose_group_ids

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaSimulationConfig

#: Product-space modalities in the order upstream draws them. Changing the
#: order changes every seeded rollout.
PRODUCT_SPACE_MODALITIES = ("bb_ca", "local_latents")

#: The backbone modality is a translation in R^3.
BACKBONE_DIM = 3


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_value = mask.unsqueeze(-1).to(x.dtype)
    count = mask_value.sum(dim=-2, keepdim=True).clamp_min(1.0)
    return (x * mask_value).sum(dim=-2, keepdim=True) / count


def force_zero_com(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove the masked center of mass along the token axis."""
    return (x - _masked_mean(x, mask)) * mask.unsqueeze(-1).to(x.dtype)


class RFProteinaProductSpaceFlowMatcher:
    """Draws RFProteína's product-space reference state.

    Args:
        config: Sampling config supplying each modality's reference settings.
        latent_dim: Local-latent width, which comes from the checkpoint rather
            than the config so the two cannot disagree.
    """

    def __init__(self, config: RFProteinaSimulationConfig, latent_dim: int) -> None:
        self.modalities: dict[str, ModalityReference] = {
            "bb_ca": ModalityReference(BACKBONE_DIM, config.bb_ca.reference),
            "local_latents": ModalityReference(latent_dim, config.local_latents.reference),
        }

    @property
    def group_keys(self) -> frozenset[str]:
        """Group tensors this configuration needs, empty when none do.

        Lets a caller skip extracting group tensors for the common scopes that
        never look at them.
        """
        keys: set[str] = set()
        for modality in self.modalities.values():
            if modality.config.stochastic_centering_scope == "per_group":
                keys.update(modality.config.stochastic_centering_group_keys)
        return frozenset(keys)

    def sample_noise(
        self,
        padding_mask: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
    ) -> RFProteinaState:
        """Draw one reference state for every product-space modality.

        Args:
            padding_mask: ``[batch, tokens]`` mask of real tokens.
            dtype: Precision of the returned state.
            generator: Random stream. ``None`` uses the default device stream.
            coordinate_condition_mask: ``[batch, tokens]``, true for tokens
                pinned to absolute coordinates. Every scope except ``all``
                leaves these at the origin.
            groups: Token-level group tensors from
                :func:`~.groups.extract_groups`, read only by the ``per_group``
                scope.

        Returns:
            One ``[batch, tokens, dim]`` tensor per modality.
        """
        padding_mask = padding_mask.to(torch.bool)
        return {
            name: self._sample_modality(
                self.modalities[name],
                padding_mask,
                dtype,
                generator,
                coordinate_condition_mask,
                groups,
            )
            for name in PRODUCT_SPACE_MODALITIES
        }

    def _sample_modality(
        self,
        modality: ModalityReference,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        coordinate_condition_mask: torch.Tensor | None,
        groups: Mapping[str, torch.Tensor | None] | None,
        apply_inference_offset: bool = True,
    ) -> torch.Tensor:
        config = modality.config
        batch, tokens = padding_mask.shape
        noise = randn((batch, tokens, modality.dim), device=padding_mask.device, dtype=dtype, generator=generator)
        noise = noise * padding_mask.unsqueeze(-1).to(dtype)
        if config.zero_com_noise:
            noise = force_zero_com(noise, padding_mask)
        # Only x_0 centering perturbs the reference draw; the other modes
        # describe training-time interpolation.
        if apply_inference_offset and config.stochastic_centering_mode == "x_0" and config.stochastic_centering_scale:
            noise = self._add_stochastic_offset(
                noise, modality, padding_mask, generator, coordinate_condition_mask, groups
            )
        return noise

    def sample_noised_backbone(
        self,
        target_backbone: torch.Tensor,
        time: torch.Tensor,
        padding_mask: torch.Tensor,
        generator: torch.Generator | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Draw the target backbone from its forward marginal at ``time``.

        Only the backbone matcher is evaluated. Sampling uses the same
        centering inputs as :meth:`sample_noise`, and applies the stochastic
        centering once, at the mode's own point of the interpolation. Inverse
        folding calls this every step so the network sees a noised copy of the
        supplied backbone rather than its own backbone update.
        """
        padding_mask = padding_mask.to(torch.bool)
        batch = padding_mask.shape[0]
        dtype = target_backbone.dtype
        if time.ndim == 0:
            time = time.expand(batch)
        time = time.to(device=padding_mask.device, dtype=dtype)
        if time.shape != (batch,):
            raise ValueError(f"time must be scalar or shape {(batch,)}, got {tuple(time.shape)}")
        if torch.any((time < 0) | (time >= 1)):
            raise ValueError("backbone resampling time must contain values in [0, 1)")

        modality = self.modalities["bb_ca"]
        expected = (batch, padding_mask.shape[1], modality.dim)
        if target_backbone.shape != expected:
            raise ValueError(f"target_backbone must have shape {expected}, got {tuple(target_backbone.shape)}")
        x_1 = target_backbone.to(device=padding_mask.device, dtype=dtype) * padding_mask.unsqueeze(-1).to(dtype)
        x_0 = self._sample_modality(
            modality,
            padding_mask,
            dtype,
            generator,
            coordinate_condition_mask,
            groups,
            apply_inference_offset=False,
        )
        mode = modality.config.stochastic_centering_mode
        if mode == "x_0" and modality.config.stochastic_centering_scale:
            x_0 = self._add_stochastic_offset(x_0, modality, padding_mask, generator, coordinate_condition_mask, groups)
        elif mode == "x_1" and modality.config.stochastic_centering_scale:
            x_1 = self._add_stochastic_offset(x_1, modality, padding_mask, generator, coordinate_condition_mask, groups)
        time_value = time.unsqueeze(-1).unsqueeze(-1)
        x_t = (1.0 - time_value) * x_0 + time_value * x_1
        if mode == "x_t" and modality.config.stochastic_centering_scale:
            x_t = self._add_stochastic_offset(x_t, modality, padding_mask, generator, coordinate_condition_mask, groups)
        return x_t * padding_mask.unsqueeze(-1).to(dtype)

    def _add_stochastic_offset(
        self,
        x: torch.Tensor,
        modality: ModalityReference,
        padding_mask: torch.Tensor,
        generator: torch.Generator | None,
        coordinate_condition_mask: torch.Tensor | None,
        groups: Mapping[str, torch.Tensor | None] | None,
    ) -> torch.Tensor:
        config = modality.config
        scope = config.stochastic_centering_scope
        if scope == "all":
            # The only scope that also moves coordinate-conditioned tokens.
            offset_mask = padding_mask
        elif coordinate_condition_mask is not None:
            offset_mask = padding_mask & coordinate_condition_mask.to(torch.bool).logical_not()
        else:
            offset_mask = padding_mask

        if scope == "per_group":
            group_ids = compose_group_ids(groups, config.stochastic_centering_group_keys)
            # Sorted unique ids keep the draw order, and therefore the seeded
            # result, independent of token layout.
            partitions = [(group_ids == group) & offset_mask for group in group_ids[offset_mask].unique()]
        else:
            partitions = [offset_mask]

        for partition in partitions:
            offset = randn((x.shape[0], x.shape[-1]), device=x.device, dtype=x.dtype, generator=generator)
            x = x + (config.stochastic_centering_scale * offset.unsqueeze(-2) * partition.unsqueeze(-1).to(x.dtype))
        return x * padding_mask.unsqueeze(-1).to(x.dtype)
