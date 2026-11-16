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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import FlowMatchingRolloutPlan, FlowMatchingSchedule, FlowMatchingState
from bionemo_ir._torch.sampling.runner import GenerativeRunner

from .contracts import (
    RFProteinaPredictFn,
    RFProteinaSimulationOutput,
    RFProteinaState,
)
from .flow_matcher import PRODUCT_SPACE_MODALITIES, RFProteinaProductSpaceFlowMatcher
from .rollout import RFProteinaFlowMatchingIntegrator

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaSimulationConfig


class RFProteinaProductSpaceSimulator:
    """Integrate supplied states and grids without choosing a generation policy."""

    def __init__(self, config: RFProteinaSimulationConfig, latent_dim: int) -> None:
        self.config = config
        self.flow_matcher = RFProteinaProductSpaceFlowMatcher(config, latent_dim)

    @property
    def group_keys(self) -> frozenset[str]:
        """Group tensors this configuration needs, empty when none do."""
        return self.flow_matcher.group_keys

    @torch.no_grad()
    def partial_simulation(
        self,
        predict: RFProteinaPredictFn,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
        initial_state: RFProteinaState,
        *,
        start_step: int = 0,
        end_step: int | None = None,
        initial_clean: RFProteinaState | None = None,
        carry_self_condition: bool,
        schedules: Mapping[str, FlowMatchingSchedule],
        recycle_depths: Sequence[int],
        can_be_encoded: torch.Tensor | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        target_backbone: torch.Tensor | None = None,
        save_trajectory_every: int = 0,
        context: SamplingContext | None = None,
        seed: int | None = None,
    ) -> RFProteinaSimulationOutput:
        """Advance caller-owned tensors over absolute intervals of the full schedule.

        The caller supplies initial state, grids and recycle depths. Clone inputs
        once here so a prediction cannot modify a sibling
        branch. The context owns both noise and feature-permutation draws.
        ``carry_self_condition`` has no default at this layer: the public model
        owns that choice, so a direct caller states whether the segment continues
        the preceding one's self-conditioning or restarts it.
        """
        if self.config.self_condition and start_step > 0 and carry_self_condition and initial_clean is None:
            raise ValueError("Resuming self-conditioning requires initial_clean from the preceding segment")
        padding_mask = padding_mask.to(torch.bool)
        initial_value = {key: initial_state[key].clone() for key in PRODUCT_SPACE_MODALITIES}
        with SamplingContext.resolve(padding_mask.device, context=context, seed=seed) as context:
            plan = FlowMatchingRolloutPlan(
                schedule=schedules,
                device=padding_mask.device,
                dtype=dtype,
            )
            if len(recycle_depths) != plan.num_steps or any(depth < 0 for depth in recycle_depths):
                raise ValueError("recycle_depths must contain one non-negative depth per schedule interval")
            resume_state = None
            if initial_clean is not None:
                clean = {modality: tensor.clone() for modality, tensor in initial_clean.items()}
                resume_state = FlowMatchingState(value=initial_value, clean=clean)
            elif start_step > 0:
                resume_state = FlowMatchingState.initial(initial_value)
            integrator = RFProteinaFlowMatchingIntegrator(
                config=self.config,
                initial_value=initial_value,
                padding_mask=padding_mask,
                can_be_encoded=can_be_encoded,
                save_trajectory_every=save_trajectory_every,
                target_backbone=target_backbone,
                flow_matcher=None if target_backbone is None else self.flow_matcher,
                coordinate_condition_mask=coordinate_condition_mask,
                groups=groups,
                resume_state=resume_state,
                segment_start_step=start_step,
                carry_self_condition=carry_self_condition,
            )

            insertion_logits = None

            def predict_velocities(model_input, time) -> dict[str, torch.Tensor]:
                nonlocal insertion_logits
                prediction = predict(
                    model_input.state,
                    time,
                    model_input.self_conditioning,
                    model_input.use_self_conditioning,
                    recycle_depths[context.step_index],
                )
                insertion = prediction.get("insertion")
                if insertion is not None:
                    insertion_logits = insertion.get("logits")
                sequence = prediction.get("sequence")
                if sequence is not None:
                    integrator.sequence_logits = sequence.get("logits")
                return {modality: prediction[modality]["v"] for modality in PRODUCT_SPACE_MODALITIES}

            result = GenerativeRunner().run_segment(
                integrator,
                plan,
                predict_velocities,
                context,
                start_step=start_step,
                end_step=end_step,
                initial_state=resume_state,
            )
            return RFProteinaSimulationOutput(
                state=result.final_state.value,
                clean_prediction=result.final_state.clean,
                insertion_logits=insertion_logits,
                trunk_sequence_logits=integrator.sequence_logits,
                trajectory=integrator.trajectory,
                seed=result.seed,
            )
