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
"""Torch-native iterative sampling primitives."""

from .contracts import (
    SamplingContext,
    SamplingIntegratorConfig,
    SamplingResult,
    SamplingRolloutPlan,
    SamplingScheduleConfig,
    StepIntegrator,
)
from .denoise_integrator import DenoiseIntegratorTemplate
from .edm import (
    AF3EDMIntegrator,
    EDMDenoiseStep,
    EDMIntegratorConfig,
    EDMRolloutPlan,
    EDMScheduleConfig,
    create_edm_schedule,
    edm_churn,
    edm_euler_update,
)
from .flow_matching import (
    FlowMatchingIntegrator,
    FlowMatchingIntegratorConfig,
    FlowMatchingModelInput,
    FlowMatchingRolloutPlan,
    FlowMatchingSchedule,
    FlowMatchingState,
    FlowMatchingStep,
    flow_euler_update,
    velocity_to_clean_endpoint,
)
from .hooks import AfterDenoiseHook, BeforeDenoiseHook, DenoiseHookPipeline
from .runner import GenerativeRunner, SamplingCancelled

__all__ = [
    "AF3EDMIntegrator",
    "AfterDenoiseHook",
    "BeforeDenoiseHook",
    "DenoiseHookPipeline",
    "DenoiseIntegratorTemplate",
    "EDMDenoiseStep",
    "EDMIntegratorConfig",
    "EDMRolloutPlan",
    "EDMScheduleConfig",
    "FlowMatchingIntegrator",
    "FlowMatchingIntegratorConfig",
    "FlowMatchingModelInput",
    "FlowMatchingRolloutPlan",
    "FlowMatchingSchedule",
    "FlowMatchingState",
    "FlowMatchingStep",
    "GenerativeRunner",
    "SamplingCancelled",
    "SamplingContext",
    "SamplingIntegratorConfig",
    "SamplingResult",
    "SamplingRolloutPlan",
    "SamplingScheduleConfig",
    "StepIntegrator",
    "create_edm_schedule",
    "edm_churn",
    "edm_euler_update",
    "flow_euler_update",
    "velocity_to_clean_endpoint",
]
