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
        state = integrator.initialize(plan, context)
        num_steps = integrator.num_steps(plan)
        for step_index in range(num_steps):
            if context.is_cancelled is not None and context.is_cancelled():
                raise SamplingCancelled(f"Sampling cancelled before step {step_index}")
            context.step_index = step_index
            state = integrator.step(step_index, state, plan, predict, context)
        return SamplingResult(final_state=state, seed=context.seed, num_steps=num_steps)
