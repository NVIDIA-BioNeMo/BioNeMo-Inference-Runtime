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
"""Flow-matching schedules and integration primitives.

Flow matching transports a reference sample to a data sample along a
time-parameterized path. For the linear (conditional optimal-transport)
interpolant used by the protein models here,

``x_t = t * x_1 + (1 - t) * x_0``,

the network predicts the velocity ``v = dx_t/dt``, one interval is an Euler
step ``x_{t+dt} = x_t + v * dt``, and the clean endpoint implied by a velocity
is ``x_1 = x_t + (1 - t) * v``. Time increases from the reference distribution
at ``t ~ 0`` toward data at ``t = 1``, which is the opposite direction to the
decreasing noise levels of :mod:`~bionemo_ir._torch.sampling.edm`.

The pieces mirror ``edm.py`` one for one so both algorithms present the same
shape to :class:`~bionemo_ir._torch.sampling.runner.GenerativeRunner`:

- :class:`FlowMatchingSchedule` / :class:`FlowMatchingRolloutPlan` resolve the
  time grid and optional stochastic forcing, per modality;
- :class:`FlowMatchingIntegrator` implements deterministic product-space Euler
  updates through the invariant prepare/hooks/predict/hooks/complete ordering.

Sampling is *product-space*: a model may evolve several modalities on their own
time grids but through one shared network evaluation, so states, times and
velocities are all keyed by modality name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from bionemo_ir._torch.sampling.contracts import (
    SamplingContext,
    SamplingIntegratorConfig,
    SamplingRolloutPlan,
)
from bionemo_ir._torch.sampling.denoise_integrator import DenoiseIntegratorTemplate
from bionemo_ir._torch.sampling.hooks import DenoiseHookPipeline

#: Per-modality tensors: modality name -> tensor.
ModalityTensors = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FlowMatchingSchedule:
    """One monotonic time grid and optional per-interval stochastic forcing."""

    times: torch.Tensor
    noise_rate: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.times.numel() < 2:
            raise ValueError("times must be one-dimensional with at least two points")
        if not bool(torch.isfinite(self.times).all()):
            raise ValueError("times must contain only finite values")

        intervals = self.times[1:] - self.times[:-1]
        is_increasing = bool((intervals > 0).all())
        is_decreasing = bool((intervals < 0).all())
        if not (is_increasing or is_decreasing):
            raise ValueError("flow-matching times must be strictly monotonic")

        if self.noise_rate is None:
            return
        if self.noise_rate.shape != (self.num_steps,):
            raise ValueError("noise_rate must contain one value per interval")
        if self.noise_rate.device != self.times.device:
            raise ValueError("times and noise_rate devices must match")
        if not bool(torch.isfinite(self.noise_rate).all()):
            raise ValueError("noise_rate must contain only finite values")
        if not bool((self.noise_rate >= 0).all()):
            raise ValueError("noise_rate must be nonnegative")

    @property
    def num_steps(self) -> int:
        return self.times.numel() - 1


@dataclass(frozen=True)
class FlowMatchingRolloutPlan(SamplingRolloutPlan[Mapping[str, FlowMatchingSchedule]]):
    """Resolved schedules for a single- or multi-modal flow rollout."""

    schedule: Mapping[str, FlowMatchingSchedule]
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if not self.schedule:
            raise ValueError("flow-matching schedule must not be empty")

        expected_steps = None
        for modality, modality_schedule in self.schedule.items():
            if not modality:
                raise ValueError("flow-matching modality names must not be empty")
            if modality_schedule.times.device != self.device:
                raise ValueError(f"schedule for modality {modality!r} is on the wrong device")
            if expected_steps is None:
                expected_steps = modality_schedule.num_steps
            elif modality_schedule.num_steps != expected_steps:
                raise ValueError("all flow-matching modalities must use the same number of steps")

    @property
    def num_steps(self) -> int:
        return next(iter(self.schedule.values())).num_steps

    @property
    def modalities(self) -> tuple[str, ...]:
        """Modality names in schedule order."""
        return tuple(self.schedule)


@dataclass(frozen=True)
class FlowMatchingIntegratorConfig(SamplingIntegratorConfig):
    """Parameters shared by flow-matching integration methods.

    ``step_scale`` multiplies the Euler update, mirroring the same knob in
    :class:`~bionemo_ir._torch.sampling.edm.EDMIntegratorConfig`.
    ``self_condition`` feeds the previous step's predicted clean endpoint back
    into the network; it is inactive on the first step, where no prediction
    exists yet.
    """

    step_scale: float = 1.0
    self_condition: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not math.isfinite(self.step_scale):
            raise ValueError("step_scale must be finite")
        if self.step_scale <= 0:
            raise ValueError("step_scale must be positive")


@dataclass
class FlowMatchingState:
    """Persistent rollout state carried between flow-matching intervals.

    ``clean`` holds the endpoint implied by the most recent velocity, which is
    both a rollout output and the self-conditioning input of the next step. It
    is zero-filled before the first prediction.
    """

    value: dict[str, torch.Tensor]
    clean: dict[str, torch.Tensor]

    @classmethod
    def initial(cls, value: Mapping[str, torch.Tensor]) -> FlowMatchingState:
        """Wrap a starting point with a zero-filled clean prediction."""
        return cls(
            value=dict(value),
            clean={modality: torch.zeros_like(tensor) for modality, tensor in value.items()},
        )


@dataclass(frozen=True)
class FlowMatchingModelInput:
    """Everything the velocity network reads, besides the time grid."""

    state: Mapping[str, torch.Tensor]
    self_conditioning: Mapping[str, torch.Tensor]
    use_self_conditioning: bool


@dataclass
class FlowMatchingStep:
    """Resolved state for one flow-matching interval.

    ``time`` is broadcast per modality to the batch shape the network expects,
    while ``delta_time`` and ``noise_rate`` stay scalar per modality.
    """

    step_index: int
    state: dict[str, torch.Tensor]
    time: dict[str, torch.Tensor]
    delta_time: dict[str, torch.Tensor]
    noise_rate: dict[str, torch.Tensor]
    self_conditioning: dict[str, torch.Tensor]
    use_self_conditioning: bool
    velocity: dict[str, torch.Tensor] = field(default_factory=dict)
    clean: dict[str, torch.Tensor] = field(default_factory=dict)


def velocity_to_clean_endpoint(state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Recover ``x_1`` from a velocity on the linear interpolant.

    Differentiating ``x_t = t * x_1 + (1 - t) * x_0`` gives ``v = x_1 - x_0``,
    so ``x_1 = x_t + (1 - t) * v``. ``time`` is a per-batch vector broadcast
    over the trailing token and channel axes.
    """
    return state + (1.0 - time[..., None, None]) * velocity


def flow_euler_update(
    state: torch.Tensor, velocity: torch.Tensor, delta_time: torch.Tensor, step_scale: float
) -> torch.Tensor:
    """Advance one interval with ``x_{t+dt} = x_t + step_scale * v * dt``."""
    return state + step_scale * velocity * delta_time


class FlowMatchingIntegrator(
    DenoiseIntegratorTemplate[
        FlowMatchingRolloutPlan,
        FlowMatchingState,
        FlowMatchingStep,
        FlowMatchingModelInput,
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
    ]
):
    """Deterministic Euler over a dict of per-modality tensors.

    ``DenoiseIntegratorTemplate`` fixes execution order. ``initial_value``
    supplies the model-specific reference sample and predictions are
    ``{modality: velocity}``.
    """

    def __init__(
        self,
        config: FlowMatchingIntegratorConfig,
        hook_pipeline: DenoiseHookPipeline[FlowMatchingStep] | None = None,
        initial_value: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(hook_pipeline)
        self.config = config
        if initial_value is None:
            raise ValueError("initial_value is required")
        self.initial_value = initial_value

    def num_steps(self, plan: FlowMatchingRolloutPlan) -> int:
        return plan.num_steps

    def initialize(self, plan: FlowMatchingRolloutPlan, context: SamplingContext) -> FlowMatchingState:
        del plan, context
        return FlowMatchingState.initial(self.initial_value)

    def prepare_denoise_step(
        self,
        step_index: int,
        state: FlowMatchingState,
        plan: FlowMatchingRolloutPlan,
        context: SamplingContext,
    ) -> FlowMatchingStep:
        del context
        batch = next(iter(state.value.values())).shape[0]
        time, delta_time, noise_rate = {}, {}, {}
        for modality, schedule in plan.schedule.items():
            time[modality] = schedule.times[step_index].expand(batch)
            delta_time[modality] = schedule.times[step_index + 1] - schedule.times[step_index]
            noise_rate[modality] = (
                schedule.times.new_zeros(()) if schedule.noise_rate is None else schedule.noise_rate[step_index]
            )
        return FlowMatchingStep(
            step_index=step_index,
            state=dict(state.value),
            time=time,
            delta_time=delta_time,
            noise_rate=noise_rate,
            self_conditioning=dict(state.clean),
            use_self_conditioning=self.config.self_condition and step_index > 0,
        )

    def denoiser_inputs(self, step: FlowMatchingStep) -> tuple[FlowMatchingModelInput, Mapping[str, torch.Tensor]]:
        model_input = FlowMatchingModelInput(
            state=step.state,
            self_conditioning=step.self_conditioning,
            use_self_conditioning=step.use_self_conditioning,
        )
        return model_input, step.time

    def attach_prediction(self, step: FlowMatchingStep, prediction: Mapping[str, torch.Tensor]) -> FlowMatchingStep:
        step.velocity = {modality: prediction[modality] for modality in step.state}
        step.clean = {
            modality: velocity_to_clean_endpoint(step.state[modality], velocity, step.time[modality])
            for modality, velocity in step.velocity.items()
        }
        return step

    def complete_step(self, step: FlowMatchingStep) -> FlowMatchingState:
        if not step.velocity:
            raise RuntimeError("flow-matching step is missing its velocity prediction")
        value = {
            modality: flow_euler_update(
                state, step.velocity[modality], step.delta_time[modality], self.config.step_scale
            )
            for modality, state in step.state.items()
        }
        return FlowMatchingState(value=value, clean=dict(step.clean))
