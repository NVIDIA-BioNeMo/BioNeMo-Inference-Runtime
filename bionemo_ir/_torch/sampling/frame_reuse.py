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
"""Frame-aware reuse of denoiser predictions across adjacent late no-churn EDM steps.

An approximation of the denoiser, not of the integration: every EDM update, augmentation and churn
draw of the original rollout still happens, so the random streams are unchanged. At a reused step
the previous prediction is mapped through that step's rigid augmentation instead of running the
network. Only Cartesian coordinates are mapped; learned feature channels are never reused.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
from pydantic import AfterValidator

from bionemo_ir._torch.layers.random_augmentation import RigidFrame, centre_random_augmentation_with_frame
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.edm import AF3EDMIntegrator, EDMIntegratorConfig, EDMRolloutPlan

__all__ = [
    "FRAME_REUSE_AUTO",
    "FRAME_REUSE_EVALUATED_PERIODS",
    "FRAME_REUSE_EVALUATED_STEPS",
    "FRESH_TAIL_STEPS",
    "FrameReuseEDMIntegrator",
    "FrameReusePeriod",
    "FrameReuseSchedule",
    "FrameReuseSetting",
    "frame_reuse_schedule",
    "require_evaluated_period",
    "require_evaluated_steps",
    "require_forced_reuse",
    "wrap_for_frame_reuse",
]

# Step counts the derived window was evaluated with: "auto" runs other counts exact, a forced period rejects them.
FRAME_REUSE_EVALUATED_STEPS = (100, 150, 200, 400)
# Periods the option was evaluated with; others are rejected, while 0 (off) and "auto" are always allowed.
FRAME_REUSE_EVALUATED_PERIODS = (4,)
# The last steps of a rollout always run the network.
FRESH_TAIL_STEPS = 2
# Reuse with the evaluated period wherever the rollout was evaluated, and run exact elsewhere.
FRAME_REUSE_AUTO = "auto"
# Why an AF3-style plan cannot reuse: the mapping goes through the step's augmentation.
_NO_AUGMENTATION = "Frame reuse maps predictions through the step's augmentation; it is off in this plan"

type FrameReusePeriod = int | Literal["auto"]


def require_evaluated_steps(num_steps: int, evaluated_steps: Sequence[int] = FRAME_REUSE_EVALUATED_STEPS) -> None:
    """Reject a step count the derived window was not evaluated with, before any rollout work starts."""
    if num_steps not in evaluated_steps:
        raise ValueError(
            f"frame_reuse_period was only evaluated with {tuple(evaluated_steps)} sampling steps; got {num_steps}"
        )


def require_forced_reuse(
    setting: FrameReusePeriod,
    num_steps: int,
    *,
    blocker: str | None = None,
    evaluated_steps: Sequence[int] = FRAME_REUSE_EVALUATED_STEPS,
) -> None:
    """Reject a forced period, neither 0 nor ``"auto"``, that cannot reuse in a rollout of ``num_steps``.

    Models run this before the trunk with their call arguments, so a forced period fails before any
    model work instead of when the rollout starts.

    Args:
        setting: ``frame_reuse_period`` of the sampling config.
        num_steps: The rollout's step count.
        blocker: Why the sampler cannot reuse with these arguments, or None when it can.
        evaluated_steps: Step counts the model kept its accuracy at.
    """
    if setting in (0, FRAME_REUSE_AUTO):
        return
    if blocker is not None:
        raise ValueError(blocker)
    require_evaluated_steps(num_steps, evaluated_steps)


def require_evaluated_period(period: FrameReusePeriod) -> FrameReusePeriod:
    """Return ``period`` when it is ``"auto"``, 0 (off) or one the option was evaluated with, else raise."""
    if period == FRAME_REUSE_AUTO:
        return period
    if period != 0 and period not in FRAME_REUSE_EVALUATED_PERIODS:
        raise ValueError(
            f"frame_reuse_period must be 0 (off) or one of {FRAME_REUSE_EVALUATED_PERIODS}, the evaluated periods; got {period}"
        )
    return period


# The type of a sampling config's ``frame_reuse_period`` field: "auto", 0 or an evaluated period.
FrameReuseSetting = Annotated[FrameReusePeriod, AfterValidator(require_evaluated_period)]


def frame_reuse_schedule(
    setting: FrameReusePeriod,
    plan: EDMRolloutPlan,
    *,
    training: bool,
    blocker: str | None = None,
    evaluated_steps: Sequence[int] = FRAME_REUSE_EVALUATED_STEPS,
) -> FrameReuseSchedule | None:
    """The reuse schedule of one rollout, or None to run it exact.

    A rollout cannot reuse in training, under autograd, or when ``blocker`` names a reason. ``"auto"``
    then runs exact, as it does at a step count outside ``evaluated_steps`` or when the churn gate
    leaves no window before the fresh tail. A forced period raises a ``ValueError`` in each of those
    cases instead. A plan without prepared EDM levels raises even for ``"auto"``: every sampler
    prepares them, so a missing set is a sampler bug, not a configuration to run exact.

    Args:
        setting: ``frame_reuse_period`` of the sampling config.
        plan: The rollout plan; its churn gate decides the window.
        training: Whether the sampler is in training mode.
        blocker: Why the sampler cannot reuse in this rollout, such as an option that reads or edits
            every fresh prediction, or None when it can.
        evaluated_steps: Step counts the model kept its accuracy at.
    """
    if setting == 0:
        return None
    if training or torch.is_grad_enabled():
        blocker = "Frame prediction reuse is inference-only"
    if setting != FRAME_REUSE_AUTO:
        if blocker is not None:
            raise ValueError(blocker)
        return FrameReuseSchedule.for_plan(setting, plan, evaluated_steps)
    if blocker is not None or plan.num_steps not in evaluated_steps:
        return None
    window = _reuse_window(plan)
    return None if window is None else FrameReuseSchedule(FRAME_REUSE_EVALUATED_PERIODS[0], *window)


def _reuse_window(plan: EDMRolloutPlan) -> tuple[int, int] | None:
    """The plan's first churn-free step and its last reusable step, or None when no step lies between them."""
    levels = plan.churn_levels
    if levels is None:
        raise ValueError("Frame reuse needs the prepared EDM levels of the plan")
    churn_free = (levels.gamma_host == 0).nonzero()
    end = plan.num_steps - 1 - FRESH_TAIL_STEPS
    if churn_free.numel() == 0 or int(churn_free[0]) > end:
        return None
    return int(churn_free[0]), end


@dataclass(frozen=True)
class FrameReuseSchedule:
    """Which steps consume a frame-mapped prediction instead of a network evaluation.

    From ``start`` through ``end`` a step evaluates the network when it is a multiple of ``period``
    after ``start`` and reuses the previous prediction otherwise; the steps outside stay fresh.
    :meth:`for_plan` derives the window from a rollout plan: it starts at the first step without
    churn noise and leaves the last ``FRESH_TAIL_STEPS`` steps fresh. With 200 steps that is steps
    160 through 197, so 172 steps evaluate the network and 28 reuse.
    """

    period: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.period < 2 or not 0 <= self.start <= self.end:
            raise ValueError("Frame reuse needs period >= 2 and 0 <= start <= end")

    @classmethod
    def for_plan(
        cls, period: int, plan: EDMRolloutPlan, evaluated_steps: Sequence[int] = FRAME_REUSE_EVALUATED_STEPS
    ) -> FrameReuseSchedule:
        """Derive the window from the plan's noise schedule and churn gate.

        The start is the first step whose churn rate is zero (a reused step must not inject churn noise);
        the end leaves ``FRESH_TAIL_STEPS`` fresh steps. Step counts outside ``evaluated_steps`` and
        periods the option was not evaluated with are rejected here, before any rollout work.
        """
        require_evaluated_steps(plan.num_steps, evaluated_steps)
        require_evaluated_period(period)
        window = _reuse_window(plan)
        if window is None:
            raise ValueError("Frame reuse needs a churn-free step before the fresh final steps; this plan has none")
        return cls(period, *window)

    def reuses(self, step: int) -> bool:
        """Whether ``step`` takes a mapped prediction."""
        return self.start <= step <= self.end and (step - self.start) % self.period != 0

    def validate(self, plan: EDMRolloutPlan, config: EDMIntegratorConfig | None = None) -> None:
        """Require the plan to be the no-churn tail the mapping is defined for.

        A reused step must not inject churn noise, and the plan needs a step after the last reused one.
        With ``config``, the plan's levels must be prepared for it, so the churn checked here is the
        churn the integrator applies. The window was only evaluated with the step counts of
        ``FRAME_REUSE_EVALUATED_STEPS``; other counts are rejected up front.
        """
        require_evaluated_steps(plan.num_steps)
        levels = plan.churn_levels
        if levels is None:
            raise ValueError("Frame reuse needs the prepared EDM levels of the plan")
        if config is not None and levels.config != config:
            raise ValueError("Frame reuse needs the plan's churn levels to be prepared for this integrator's config")
        if plan.num_steps <= self.end + 1:
            raise ValueError("Frame reuse needs a fresh step after its last reused step")
        for step in range(plan.num_steps):
            if self.reuses(step) and float(levels.gamma_host[step]) != 0.0:
                raise ValueError(f"Step {step} injects churn noise and cannot reuse a prediction")


class FrameReuseEDMIntegrator(AF3EDMIntegrator):
    """The AF3 EDM integrator with the frame-aware prediction cache of one rollout.

    Args:
        parent: The integrator whose configuration and hooks are kept.
        schedule: Reuse schedule.
    """

    def __init__(self, parent: AF3EDMIntegrator, schedule: FrameReuseSchedule) -> None:
        super().__init__(parent.config, parent.hook_pipeline)
        self.schedule = schedule
        self.current_step = -1
        self.fresh_calls = 0
        self.reused_calls = 0
        self.fresh_steps: list[int] = []
        self._frame: RigidFrame | None = None
        self._cache: torch.Tensor | None = None

    def initialize(self, plan: EDMRolloutPlan, context: SamplingContext) -> torch.Tensor:
        if not plan.augment_coordinates:
            raise ValueError(_NO_AUGMENTATION)
        self.schedule.validate(plan, self.config)
        self.current_step, self.fresh_calls, self.reused_calls = -1, 0, 0
        self.fresh_steps = []
        self._frame = self._cache = None
        return super().initialize(plan, context)

    def augment_state(
        self, step_index: int, state: torch.Tensor, plan: EDMRolloutPlan, context: SamplingContext
    ) -> torch.Tensor:
        if step_index != self.current_step + 1:
            raise RuntimeError("Frame reuse needs the steps of one rollout in order")
        self.current_step = step_index
        state, self._frame = centre_random_augmentation_with_frame(state, plan.atom_mask, generator=context.generator)
        return state

    def predict(self, x_noisy: torch.Tensor, fresh: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Evaluate the network, or map the previous prediction into the current frame.

        A mapped prediction is cached again for the next step, so the cache always holds the latest
        prediction in the frame the current state lives in.

        Args:
            x_noisy: The step's noisy coordinates, only checked against the cache.
            fresh: Runs the network for this step.
        """
        if self.schedule.reuses(self.current_step):
            if self._cache is None or self._cache.shape != x_noisy.shape:
                raise RuntimeError("No cached prediction to map into the current frame")
            prediction = self._frame.apply(self._cache)
            self.reused_calls += 1
        else:
            prediction = fresh()
            self.fresh_calls += 1
            self.fresh_steps.append(self.current_step)
        self._cache = prediction.detach().clone() if self.schedule.reuses(self.current_step + 1) else None
        self._frame = None
        return prediction

    def bind(self, evaluate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Callable[..., torch.Tensor]:
        """The ``predict(x_noisy, sigma_hat)`` closure of a sampler whose network call is ``evaluate``."""
        return lambda x_noisy, sigma_hat: self.predict(x_noisy, lambda: evaluate(x_noisy, sigma_hat))

    @property
    def receipt(self) -> dict[str, object]:
        """Counts of the last rollout: network evaluations, mapped predictions and the fresh steps."""
        return {"fresh": self.fresh_calls, "reused": self.reused_calls, "fresh_steps": list(self.fresh_steps)}


def wrap_for_frame_reuse(
    parent: AF3EDMIntegrator,
    setting: FrameReusePeriod,
    plan: EDMRolloutPlan,
    evaluate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    training: bool,
) -> tuple[AF3EDMIntegrator, Callable[..., torch.Tensor]]:
    """The integrator of one rollout of an AF3-style sampler and its ``predict`` closure.

    Returns ``parent`` and ``evaluate`` unchanged when :func:`frame_reuse_schedule` runs the rollout exact.

    Args:
        parent: The sampler's integrator.
        setting: ``frame_reuse_period`` of the sampling config.
        plan: The rollout plan; the window is derived from it.
        evaluate: The sampler's network call, ``evaluate(x_noisy, sigma_hat)``.
        training: Whether the sampler is in training mode; reuse is inference-only.
    """
    blocker = None if plan.augment_coordinates else _NO_AUGMENTATION
    schedule = frame_reuse_schedule(setting, plan, training=training, blocker=blocker)
    if schedule is None:
        return parent, evaluate
    integrator = FrameReuseEDMIntegrator(parent, schedule)
    return integrator, integrator.bind(evaluate)
