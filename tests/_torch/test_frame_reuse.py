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
"""Frame-aware prediction reuse: schedule, rollout counts, random streams and the mapped prediction."""

import dataclasses

import pytest
import torch

from bionemo_ir._torch.layers.random_augmentation import centre_random_augmentation
from bionemo_ir._torch.sampling import AF3EDMIntegrator, EDMIntegratorConfig, EDMRolloutPlan, GenerativeRunner
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.frame_reuse import FrameReuseEDMIntegrator, FrameReuseSchedule

_STEPS = 200


def _plan(device: str, atoms: int = 17) -> EDMRolloutPlan:
    schedule = torch.cat([torch.logspace(2.2, -3.4, _STEPS, device=device), torch.zeros(1, device=device)])
    return EDMRolloutPlan(
        schedule, (1, 5, atoms, 3), schedule.device, torch.float32, integrator_config=EDMIntegratorConfig()
    )


def _denoiser(x_noisy: torch.Tensor, sigma_hat: torch.Tensor) -> torch.Tensor:
    # Rotation- and translation-equivariant: shrink toward the centroid.
    return 0.9 * x_noisy + 0.1 * x_noisy.mean(dim=-2, keepdim=True)


def test_schedule_counts_and_steps() -> None:
    schedule = FrameReuseSchedule(period=4)
    reused = [step for step in range(_STEPS) if schedule.reuses(step)]
    assert len(reused) == 28 and _STEPS - len(reused) == 172
    assert not any(schedule.reuses(step) for step in (*range(160), 160, 164, 196, 198, 199))
    assert all(step % 4 != 0 for step in reused) and min(reused) == 161 and max(reused) == 197


@pytest.mark.parametrize("period, start, end", [(1, 160, 197), (4, 200, 199)])
def test_schedule_rejects_degenerate_policies(period: int, start: int, end: int) -> None:
    with pytest.raises(ValueError):
        FrameReuseSchedule(period=period, start=start, end=end)


def test_other_step_counts_are_rejected_up_front() -> None:
    plan = EDMRolloutPlan(
        torch.cat([torch.logspace(2.2, -3.4, 100), torch.zeros(1)]),
        (1, 5, 17, 3),
        torch.device("cpu"),
        torch.float32,
        integrator_config=EDMIntegratorConfig(),
    )
    with pytest.raises(ValueError, match="only evaluated with 200 sampling steps; got 100"):
        FrameReuseSchedule(period=4).validate(plan)


def test_reuse_needs_coordinate_augmentation() -> None:
    plan = dataclasses.replace(_plan("cpu"), augment_coordinates=False)
    with pytest.raises(ValueError, match="augmentation"):
        FrameReuseSchedule(period=4).validate(plan)


def test_plan_levels_must_match_the_integrator_config() -> None:
    plan = _plan("cpu")
    other = AF3EDMIntegrator(EDMIntegratorConfig(gamma0=0.0))
    reuse = FrameReuseEDMIntegrator(other, FrameReuseSchedule(period=4))
    with SamplingContext.graph_safe(torch.device("cpu"), 7) as context, pytest.raises(ValueError, match="config"):
        reuse.initialize(plan, context)


def test_churn_steps_cannot_reuse() -> None:
    plan = _plan("cpu")
    with pytest.raises(ValueError, match="churn"):
        FrameReuseSchedule(period=4, start=10, end=197).validate(plan)


def test_rollout_counts_random_streams_and_mapped_prediction() -> None:
    plan = _plan("cpu")
    base = AF3EDMIntegrator(EDMIntegratorConfig())
    reuse = FrameReuseEDMIntegrator(base, FrameReuseSchedule(period=4))
    runner = GenerativeRunner()
    with SamplingContext.graph_safe(torch.device("cpu"), 7) as context:
        runner.run(base, plan, _denoiser, context)
        baseline_state = context.generator.get_state()

    fresh_outputs: dict[int, torch.Tensor] = {}
    mapped: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def predict(x_noisy: torch.Tensor, sigma_hat: torch.Tensor) -> torch.Tensor:
        step = reuse.current_step
        frame = reuse._frame
        previous = reuse._cache

        def fresh() -> torch.Tensor:
            fresh_outputs[step] = _denoiser(x_noisy, sigma_hat)
            return fresh_outputs[step]

        out = reuse.predict(x_noisy, fresh)
        if step not in fresh_outputs:
            mapped[step] = (frame.apply(previous), out)
        return out

    with SamplingContext.graph_safe(torch.device("cpu"), 7) as context:
        runner.run(reuse, plan, predict, context)
        assert torch.equal(context.generator.get_state(), baseline_state)
    assert reuse.receipt["fresh"] == 172 and reuse.receipt["reused"] == 28
    assert reuse.receipt["fresh_steps"][-2:] == [198, 199] and 160 in reuse.receipt["fresh_steps"]
    assert len(mapped) == 28 and all(torch.equal(want, got) for want, got in mapped.values())
    assert reuse._cache is None


def test_frame_applies_the_augmentation() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, 5, 11, 3, generator=generator)
    mask = torch.ones(1, 11)
    mask[:, -2:] = 0
    from bionemo_ir._torch.layers.random_augmentation import centre_random_augmentation_with_frame

    state = generator.get_state()
    augmented, frame = centre_random_augmentation_with_frame(x, mask, generator=generator)
    generator.set_state(state)
    assert torch.equal(augmented, centre_random_augmentation(x, mask, generator=generator))
    assert torch.equal(frame.apply(x), augmented)


def test_frame_reuse_period_follows_the_sampling_config() -> None:
    from bionemo_ir.models.protenix.config import EDMSamplingConfig

    assert EDMSamplingConfig().frame_reuse_period == 0
    assert EDMSamplingConfig(frame_reuse_period=4).frame_reuse_period == 4
    for period in (-1, 1):
        with pytest.raises(ValueError):
            EDMSamplingConfig(frame_reuse_period=period)
    with pytest.raises(ValueError, match="only evaluated with 200 sampling steps; got 50"):
        EDMSamplingConfig(frame_reuse_period=4, n_step=50)
    assert EDMSamplingConfig(n_step=50).frame_reuse_period == 0
