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
"""Shared loop orchestration for Torch-native iterative samplers."""

from __future__ import annotations

from bionemo_ir._torch.sampling.contracts import (
    ModelInputT,
    PlanT,
    PredictFn,
    PredictionT,
    SamplingContext,
    SamplingResult,
    StateT,
    StepIntegrator,
    TimeT,
)


class SamplingCancelled(RuntimeError):
    """Raised when a rollout is cancelled between integration steps."""


class GenerativeRunner[PlanT, StateT, ModelInputT, TimeT, PredictionT]:
    """Own iteration while an integrator owns one mathematical step."""

    def run(
        self,
        integrator: StepIntegrator[PlanT, StateT, ModelInputT, TimeT, PredictionT],
        plan: PlanT,
        predict: PredictFn[ModelInputT, TimeT, PredictionT],
        context: SamplingContext,
    ) -> SamplingResult[StateT]:
        """Run every interval in ``plan`` from a freshly initialized state."""
        return self.run_segment(integrator, plan, predict, context)

    def run_segment(
        self,
        integrator: StepIntegrator[PlanT, StateT, ModelInputT, TimeT, PredictionT],
        plan: PlanT,
        predict: PredictFn[ModelInputT, TimeT, PredictionT],
        context: SamplingContext,
        *,
        start_step: int = 0,
        end_step: int | None = None,
        initial_state: StateT | None = None,
    ) -> SamplingResult[StateT]:
        """Integrate global steps ``[start_step, end_step)`` on the full plan.

        ``initialize`` prepares integrator-local segment state. For a
        continuation its sampled state is discarded and the generator is
        restored, so lifecycle setup cannot perturb the rollout's random
        stream. Resuming past step 0 requires ``initial_state``. Step indices
        are the plan's global indices so times and last-step logic stay correct
        across segments.

        Args:
            integrator: Mathematical step implementation.
            plan: Full rollout plan covering every interval.
            predict: Denoiser callback.
            context: Mutable rollout context, including the generator.
            start_step: Inclusive global step index.
            end_step: Exclusive global step index. ``None`` runs to the end
                of ``plan``.
            initial_state: Continuation to resume from. Required when
                ``start_step > 0``.

        Returns:
            The state after ``end_step - start_step`` integrations, with
            ``num_steps`` equal to that count.
        """
        num_steps = integrator.num_steps(plan)
        stop = num_steps if end_step is None else end_step
        if start_step < 0 or stop < start_step or stop > num_steps:
            raise ValueError(f"invalid segment [{start_step}, {stop}) for a plan of {num_steps} steps")
        if start_step != 0 and initial_state is None:
            raise ValueError("resuming a rollout requires initial_state")
        if initial_state is None:
            state = integrator.initialize(plan, context)
        else:
            generator_state = context.generator.get_state()
            try:
                integrator.initialize(plan, context)
            finally:
                context.generator.set_state(generator_state)
            state = initial_state
        for step_index in range(start_step, stop):
            context.step_index = step_index
            if context.is_cancelled is not None and context.is_cancelled():
                raise SamplingCancelled(f"Sampling cancelled before step {step_index}")
            state = integrator.step(step_index, state, plan, predict, context)
        return SamplingResult(final_state=state, seed=context.seed, num_steps=stop - start_step)
