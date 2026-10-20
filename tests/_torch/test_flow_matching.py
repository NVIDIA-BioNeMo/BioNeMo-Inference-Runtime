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

import pytest
import torch

from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import (
    FlowMatchingIntegrator,
    FlowMatchingIntegratorConfig,
    FlowMatchingRolloutPlan,
    FlowMatchingSchedule,
)
from bionemo_ir._torch.sampling.runner import GenerativeRunner, SamplingCancelled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _linear_plan(num_steps: int) -> FlowMatchingRolloutPlan:
    times = torch.linspace(0.0, 1.0, num_steps + 1, device="cuda")
    return FlowMatchingRolloutPlan(
        schedule={"x": FlowMatchingSchedule(times=times)},
        device=times.device,
        dtype=torch.float32,
    )


def test_flow_matching_integrator_integrates_constant_velocity() -> None:
    """The default integrator must reproduce the closed-form Euler trajectory.

    With a constant velocity the exact solution of ``dx/dt = v`` over ``[0, 1]``
    is ``x_0 + v``, which Euler reaches exactly regardless of step count.
    """
    start = torch.randn(2, 5, 3, device="cuda")
    velocity = torch.randn(2, 5, 3, device="cuda")
    plan = _linear_plan(num_steps=8)
    integrator = FlowMatchingIntegrator(
        config=FlowMatchingIntegratorConfig(),
        initial_value={"x": start},
    )

    result = GenerativeRunner().run(
        integrator,
        plan,
        lambda model_input, time: {"x": velocity},
        SamplingContext.create(torch.device("cuda"), seed=0),
    )

    assert result.num_steps == 8
    torch.testing.assert_close(result.final_state.value["x"], start + velocity)


def test_flow_matching_step_scale_and_self_conditioning() -> None:
    """``step_scale`` scales the update; self-conditioning lags by one step."""
    start = torch.zeros(1, 2, 3, device="cuda")
    velocity = torch.ones(1, 2, 3, device="cuda")
    plan = _linear_plan(num_steps=4)
    seen: list[tuple[bool, float]] = []

    def predict(model_input, time):
        seen.append((model_input.use_self_conditioning, model_input.self_conditioning["x"].abs().max().item()))
        return {"x": velocity}

    integrator = FlowMatchingIntegrator(
        config=FlowMatchingIntegratorConfig(step_scale=2.0, self_condition=True),
        initial_value={"x": start},
    )
    result = GenerativeRunner().run(integrator, plan, predict, SamplingContext.create(torch.device("cuda"), seed=0))

    torch.testing.assert_close(result.final_state.value["x"], 2.0 * velocity)
    assert [active for active, _ in seen] == [False, True, True, True]
    # The clean endpoint x_1 = x_t + (1 - t) v is zero-filled only before the
    # first prediction, so the very first call sees no self-conditioning.
    assert seen[0][1] == 0.0
    assert all(magnitude > 0.0 for _, magnitude in seen[1:])


def test_flow_matching_runner_honours_cancellation() -> None:
    plan = _linear_plan(num_steps=6)
    context = SamplingContext.create(torch.device("cuda"), seed=0)
    context.is_cancelled = lambda: context.step_index >= 2
    integrator = FlowMatchingIntegrator(
        config=FlowMatchingIntegratorConfig(),
        initial_value={"x": torch.zeros(1, 2, 3, device="cuda")},
    )

    with pytest.raises(SamplingCancelled):
        GenerativeRunner().run(
            integrator, plan, lambda model_input, time: {"x": torch.zeros(1, 2, 3, device="cuda")}, context
        )
