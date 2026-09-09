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
"""Template-method integrator for denoiser-driven sampling steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final

from bionemo_ir._torch.sampling.contracts import (
    ModelInputT,
    PlanT,
    PredictFn,
    PredictionT,
    SamplingContext,
    StateT,
    StepT,
    TimeT,
)
from bionemo_ir._torch.sampling.hooks import DenoiseHookPipeline


class DenoiseIntegratorTemplate[PlanT, StateT, StepT, ModelInputT, TimeT, PredictionT](ABC):
    """Fix denoiser-step ordering while subclasses define typed operations."""

    def __init__(
        self,
        hook_pipeline: DenoiseHookPipeline[StepT] | None = None,
    ) -> None:
        self.hook_pipeline = hook_pipeline if hook_pipeline is not None else DenoiseHookPipeline()

    @abstractmethod
    def num_steps(self, plan: PlanT) -> int:
        """Return the number of model evaluations in the rollout."""

    @abstractmethod
    def initialize(self, plan: PlanT, context: SamplingContext) -> StateT:
        """Construct the persistent state before the first step."""

    @final
    def step(
        self,
        step_index: int,
        state: StateT,
        plan: PlanT,
        predict: PredictFn[ModelInputT, TimeT, PredictionT],
        context: SamplingContext,
    ) -> StateT:
        """Run the invariant prepare/hooks/predict/hooks/complete pipeline."""
        denoise_step = self.prepare_denoise_step(step_index, state, plan, context)
        denoise_step = self.hook_pipeline.run_before(denoise_step, context)
        model_input, time = self.denoiser_inputs(denoise_step)
        prediction = predict(model_input, time)
        denoise_step = self.attach_prediction(denoise_step, prediction)
        denoise_step = self.hook_pipeline.run_after(denoise_step, context)
        return self.complete_step(denoise_step)

    @abstractmethod
    def prepare_denoise_step(
        self,
        step_index: int,
        state: StateT,
        plan: PlanT,
        context: SamplingContext,
    ) -> StepT:
        """Resolve one algorithm-specific step before hooks run."""

    @abstractmethod
    def denoiser_inputs(self, step: StepT) -> tuple[ModelInputT, TimeT]:
        """Extract the model input and time/noise conditioning."""

    @abstractmethod
    def attach_prediction(self, step: StepT, prediction: PredictionT) -> StepT:
        """Attach one model prediction to the typed step object."""

    @abstractmethod
    def complete_step(self, step: StepT) -> StateT:
        """Apply the mathematical state update after post-denoise hooks."""
