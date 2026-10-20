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

"""RFProteína's concrete flow-matching integrator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import (
    FlowMatchingIntegrator,
    FlowMatchingIntegratorConfig,
    FlowMatchingRolloutPlan,
    FlowMatchingState,
    FlowMatchingStep,
)

# ``RFProteinaPrediction`` is re-exported for callers that used to import the
# velocity-network return type from here; ``contracts`` owns the definition.
from .contracts import RFProteinaPrediction, RFProteinaState  # noqa: F401
from .flow_matcher import RFProteinaProductSpaceFlowMatcher
from .vf_tsr import velocity_flow_step, velocity_to_clean

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaSamplingConfig


class RFProteinaFlowMatchingIntegrator(FlowMatchingIntegrator):
    """Product-space VF-TSR rollout for the released RFProteína path.

    Reuses the default integrator's interval bookkeeping (times, deltas, noise
    rates, self-conditioning handoff) and replaces the parts that are
    model-specific:

    - clean endpoints are mask-aware;
    - the interval update is RFProteína's Euler or VF-TSR step rather than
      plain Euler, and latents are re-masked to the encodable residues;
    - the rollout records optional trajectory snapshots.
    """

    def __init__(
        self,
        config: RFProteinaSamplingConfig,
        initial_value: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        can_be_encoded: torch.Tensor | None = None,
        save_trajectory_every: int = 0,
        target_backbone: torch.Tensor | None = None,
        flow_matcher: RFProteinaProductSpaceFlowMatcher | None = None,
        coordinate_condition_mask: torch.Tensor | None = None,
        groups: Mapping[str, torch.Tensor | None] | None = None,
        resume_state: FlowMatchingState | None = None,
        segment_start_step: int = 0,
        carry_self_condition: bool = False,
    ) -> None:
        super().__init__(
            FlowMatchingIntegratorConfig(self_condition=config.self_condition),
            initial_value=initial_value,
            segment_start_step=segment_start_step,
            carry_self_condition=carry_self_condition,
        )
        if target_backbone is not None and flow_matcher is None:
            raise ValueError("inverse folding requires the product-space flow matcher to resample the backbone")
        self.sampling_config = config
        self.padding_mask = padding_mask
        self.can_be_encoded = can_be_encoded
        self.save_trajectory_every = save_trajectory_every
        self.target_backbone = None if target_backbone is None else target_backbone.clone()
        self.flow_matcher = flow_matcher
        self.coordinate_condition_mask = coordinate_condition_mask
        self.groups = groups
        self.modality_configs = {
            "bb_ca": config.backbone,
            "local_latents": config.local_latents,
        }
        self.trajectory: list[dict[str, RFProteinaState]] = []
        self._generator: torch.Generator | None = None
        self._num_steps = config.num_steps
        self._plan: FlowMatchingRolloutPlan | None = None
        self._resume_state = None if resume_state is None else resume_state.clone()

    def initialize(self, plan: FlowMatchingRolloutPlan, context: SamplingContext) -> FlowMatchingState:
        # The stochastic VF-TSR update draws inside complete_step, which the
        # template calls without the context, so capture the generator here.
        self._generator = context.generator
        self._num_steps = plan.num_steps
        self._plan = plan
        if self._resume_state is not None:
            return self._resume_state
        state = super().initialize(plan, context)
        if self.target_backbone is not None:
            state.value["bb_ca"] = self._resample_backbone(plan.schedule["bb_ca"].times[0])
        return state

    def _resample_backbone(self, time: torch.Tensor) -> torch.Tensor:
        """Draw the frozen target backbone from its forward marginal at ``time``."""
        if self.flow_matcher is None or self.target_backbone is None:
            raise RuntimeError("backbone resampling requires a flow matcher and target backbone")
        batch = self.target_backbone.shape[0]
        return self.flow_matcher.sample_noised_backbone(
            self.target_backbone,
            time.expand(batch) if time.ndim == 0 else time,
            self.padding_mask,
            generator=self._generator,
            coordinate_condition_mask=self.coordinate_condition_mask,
            groups=self.groups,
        )

    def attach_prediction(self, step: FlowMatchingStep, prediction: Mapping[str, torch.Tensor]) -> FlowMatchingStep:
        step.velocity = {modality: prediction[modality] for modality in step.state}
        step.clean = {
            modality: velocity_to_clean(step.state[modality], velocity, step.time[modality], self.padding_mask)
            for modality, velocity in step.velocity.items()
        }
        return step

    def complete_step(self, step: FlowMatchingStep) -> FlowMatchingState:
        if not step.velocity:
            raise RuntimeError("flow-matching step is missing its velocity prediction")

        value: RFProteinaState = {}
        for modality, state in step.state.items():
            # VF-TSR picks its ODE window from this step's time. Taking it off
            # the schedule's host copy keeps the update free of device reads.
            schedule = None if self._plan is None else self._plan.schedule.get(modality)
            value[modality] = velocity_flow_step(
                state,
                step.velocity[modality],
                step.time[modality],
                step.delta_time[modality],
                step.noise_rate[modality],
                self.padding_mask,
                self.modality_configs[modality].step,
                generator=self._generator,
                time_value=None if schedule is None else schedule.host_times[step.step_index],
            )
        if self.can_be_encoded is not None:
            latents = value["local_latents"]
            value["local_latents"] = latents * self.can_be_encoded.unsqueeze(-1).to(latents.dtype)

        clean = dict(step.clean)
        if self.target_backbone is not None:
            # Inverse folding: the network may not move the backbone. Self-cond
            # sees the clean target, and the state is redrawn from that target's
            # forward marginal (restored exactly on the last step).
            clean["bb_ca"] = self.target_backbone
            if step.step_index == self._num_steps - 1:
                value["bb_ca"] = self.target_backbone.clone()
            else:
                if self._plan is None:
                    raise RuntimeError("inverse folding requires the rollout plan to resample the next backbone")
                value["bb_ca"] = self._resample_backbone(self._plan.schedule["bb_ca"].times[step.step_index + 1])

        self._record(step.step_index, value, clean)
        return FlowMatchingState(value=value, clean=clean)

    def _record(self, step_index: int, value: RFProteinaState, clean: RFProteinaState) -> None:
        """Snapshot the updated state on the requested stride and last step."""
        every = self.save_trajectory_every
        if every <= 0:
            return
        if step_index % every and step_index != self._num_steps - 1:
            return
        self.trajectory.append(
            {
                "x_t": {key: tensor.detach().clone() for key, tensor in value.items()},
                "x_1": {key: tensor.detach().clone() for key, tensor in clean.items()},
            }
        )
