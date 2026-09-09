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
"""Ordered extension points around iterative denoiser calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from bionemo_ir._torch.sampling.contracts import SamplingContext, StepT


class BeforeDenoiseHook(Protocol[StepT]):
    """Transform one resolved step before its denoiser invocation."""

    def before_denoise(self, step: StepT, context: SamplingContext) -> StepT:
        """Return the step to pass to the next pre-denoise hook."""


class AfterDenoiseHook(Protocol[StepT]):
    """Transform one resolved step after its denoiser invocation."""

    def after_denoise(self, step: StepT, context: SamplingContext) -> StepT:
        """Return the step to pass to the next post-denoise hook."""


class DenoiseHookPipeline[StepT]:
    """Apply ordered hook objects on both sides of a denoiser call."""

    def __init__(
        self,
        before_denoise_hooks: Sequence[BeforeDenoiseHook[StepT]] = (),
        after_denoise_hooks: Sequence[AfterDenoiseHook[StepT]] = (),
    ) -> None:
        self.before_denoise_hooks = tuple(before_denoise_hooks)
        self.after_denoise_hooks = tuple(after_denoise_hooks)

    def run_before(self, step: StepT, context: SamplingContext) -> StepT:
        for hook in self.before_denoise_hooks:
            step = hook.before_denoise(step, context)
        return step

    def run_after(self, step: StepT, context: SamplingContext) -> StepT:
        for hook in self.after_denoise_hooks:
            step = hook.after_denoise(step, context)
        return step
