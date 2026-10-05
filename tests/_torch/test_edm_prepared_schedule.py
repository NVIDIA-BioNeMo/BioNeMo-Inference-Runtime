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

"""Verify prepared EDM levels preserve rollout behavior."""

import pytest
import torch

from bionemo_ir._torch.sampling import (
    AF3EDMIntegrator,
    EDMIntegratorConfig,
    EDMRolloutPlan,
    GenerativeRunner,
    SamplingContext,
)

_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("device", _DEVICES)
def test_prepared_levels_match(dtype: torch.dtype, device: str) -> None:
    schedule = torch.tensor([2560.0, 80.0, 1.01, 1.0, 0.01, 0.0], device=device, dtype=dtype)
    config = EDMIntegratorConfig()
    plan = EDMRolloutPlan(schedule, (1, 5, 17, 3), schedule.device, dtype, integrator_config=config)
    assert plan.churn_levels is not None
    assert plan.churn_levels.config == config
    expected = torch.stack(
        [
            schedule[i]
            * (
                torch.where(
                    schedule[i + 1] > config.gamma_min, schedule.new_tensor(config.gamma0), schedule.new_zeros(())
                )
                + 1
            )
            for i in range(plan.num_steps)
        ]
    )
    torch.testing.assert_close(plan.churn_levels.sigma_hat, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("device", _DEVICES)
def test_prepared_rollout_matches(mismatch: bool, device: str) -> None:
    device = torch.device("cuda:0" if device == "cuda" else device)
    schedule = torch.tensor([80.0, 2.0, 1.0, 0.1, 0.0], device=device)
    config = EDMIntegratorConfig()
    cached_config = EDMIntegratorConfig(gamma0=0.0) if mismatch else config
    common = {
        "schedule": schedule,
        "coords_shape": (2, 3, 17, 3),
        "device": device,
        "dtype": torch.float32,
        "atom_mask": torch.ones(2, 17, device=device),
    }
    plain = EDMRolloutPlan(**common)
    prepared = EDMRolloutPlan(**common, integrator_config=cached_config)
    integrator = AF3EDMIntegrator(config)

    def predict(x: torch.Tensor, _sigma: torch.Tensor) -> torch.Tensor:
        return x * 0.7

    with torch.inference_mode():
        context = SamplingContext.create(device, seed=42)
        expected = GenerativeRunner().run(integrator, plain, predict, context).final_state
        final_rng = context.generator.get_state()
        context = SamplingContext.create(device, seed=42)
        actual = GenerativeRunner().run(integrator, prepared, predict, context).final_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(context.generator.get_state(), final_rng)


def test_prepared_levels_are_derived() -> None:
    with pytest.raises(TypeError, match="churn_levels"):
        EDMRolloutPlan(torch.ones(2), (1, 3), torch.device("cpu"), torch.float32, churn_levels=None)
    with pytest.raises(ValueError, match="nonincreasing"):
        EDMRolloutPlan(
            torch.tensor([1.0, 2.0]),
            (1, 3),
            torch.device("cpu"),
            torch.float32,
            integrator_config=EDMIntegratorConfig(),
        )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("gamma0", [0.0, 0.8, 1e40])
def test_scalar_steps_match(device: str, dtype: torch.dtype, gamma0: float) -> None:
    schedule = torch.tensor([8, 8, 2, 1, 0.1, 0], device=device, dtype=dtype)
    config = EDMIntegratorConfig(gamma0=gamma0)
    plan = EDMRolloutPlan(schedule, (1, 3), schedule.device, dtype, integrator_config=config)
    levels = plan.churn_levels
    assert levels is not None
    for i, (last, next_, gamma) in enumerate(levels.scalar_steps):
        expected_gamma = torch.where(
            schedule[i + 1] > config.gamma_min, schedule.new_tensor(gamma0), schedule.new_zeros(())
        ).item()
        assert (last, next_, gamma) == (schedule[i].item(), schedule[i + 1].item(), expected_gamma)
    assert levels.scalar_steps is levels.scalar_steps
    torch.testing.assert_close(levels.sigma_hat[-2:], schedule[-3:-1], rtol=0, atol=0)


def test_prepared_schedule_snapshot() -> None:
    schedule = torch.tensor([8.0, 99.0, 2.0, 99.0, 0.0])[::2]
    plan = EDMRolloutPlan(schedule, (1, 3), schedule.device, schedule.dtype, integrator_config=EDMIntegratorConfig())
    expected = plan.schedule.clone()
    schedule.zero_()
    torch.testing.assert_close(plan.schedule, expected, atol=0, rtol=0)
    assert plan.churn_levels.scalar_steps[0][0] == 8.0


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("augment", [False, True])
def test_boltz_prepared_rollout(device: str, mismatch: bool, augment: bool) -> None:
    from bionemo_ir._torch.modules.boltz.structure import (
        BoltzDenoisePrediction,
        BoltzEDMChurnHook,
        BoltzEDMIntegrator,
        BoltzRandomRigidAugmentationHook,
    )
    from bionemo_ir._torch.sampling import DenoiseHookPipeline

    schedule = torch.tensor([8.0, 2.0, 1.0, 0.01, 0.0], device=device)
    config = EDMIntegratorConfig()
    prepared_config = EDMIntegratorConfig(gamma0=0.0) if mismatch else config
    hooks = [BoltzRandomRigidAugmentationHook(2, 3)] if augment else []
    hooks.append(BoltzEDMChurnHook(config.noise_scale))
    integrator = BoltzEDMIntegrator(config, DenoiseHookPipeline(before_denoise_hooks=tuple(hooks)))
    common = {
        "schedule": schedule,
        "coords_shape": (2, 3, 17, 3),
        "device": schedule.device,
        "dtype": torch.float32,
        "atom_mask": torch.ones(2, 3, 17, device=device),
        "augment_coordinates": False,
    }
    plain = EDMRolloutPlan(**common)
    prepared = EDMRolloutPlan(**common, integrator_config=prepared_config)

    def predict(x: torch.Tensor, _sigma: float) -> BoltzDenoisePrediction:
        return BoltzDenoisePrediction(x * 0.7, None)

    with torch.inference_mode():
        context = SamplingContext.create(schedule.device, seed=42)
        expected = GenerativeRunner().run(integrator, plain, predict, context).final_state.atom_coords
        rng = context.generator.get_state()
        context = SamplingContext.create(schedule.device, seed=42)
        actual = GenerativeRunner().run(integrator, prepared, predict, context).final_state.atom_coords
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(context.generator.get_state(), rng)


@pytest.mark.parametrize("final", ["keep", "zero", "append_zero"])
@pytest.mark.parametrize("num_steps", [1, 2, 200])
def test_prepared_terminal_modes(final: str, num_steps: int) -> None:
    from bionemo_ir._torch.sampling import EDMScheduleConfig

    schedule = EDMScheduleConfig(sigma_data=16, s_max=160, s_min=0.0004, rho=7, final=final).build(num_steps)
    config = EDMIntegratorConfig()
    plan = EDMRolloutPlan(schedule, (1, 3), schedule.device, schedule.dtype, integrator_config=config)
    assert plan.churn_levels.sigma_hat.shape == (num_steps,)
    assert len(plan.churn_levels.scalar_steps) == num_steps
    for i, (last, next_, gamma) in enumerate(plan.churn_levels.scalar_steps):
        assert last == schedule[i].item()
        assert next_ == schedule[i + 1].item()
        assert gamma == config.churn_rates(schedule[i + 1]).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prepared_graph_replay() -> None:
    device = torch.device("cuda:0")
    schedule = torch.tensor([8.0, 2.0, 0.0], device=device)
    config = EDMIntegratorConfig()
    plan = EDMRolloutPlan(
        schedule,
        (2, 3, 17, 3),
        device,
        torch.float32,
        augment_coordinates=False,
        integrator_config=config,
    )
    integrator = AF3EDMIntegrator(config)
    state = torch.ones(plan.coords_shape, device=device)
    context = SamplingContext.create(device, seed=42)

    def predict(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x / (sigma + 1)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        integrator.step(0, state, plan, predict, context)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(context.generator)
    with torch.cuda.graph(graph):
        captured = integrator.step(0, state, plan, predict, context)
    context.generator.manual_seed(42)
    eager_context = SamplingContext.create(device, seed=42)
    for _ in range(3):
        graph.replay()
        expected = integrator.step(0, state, plan, predict, eager_context)
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_host_schedule_moves_to_device(device: str) -> None:
    """A CPU-built schedule is validated on the host and lands on the rollout device."""
    device = torch.device("cuda:0" if device == "cuda" else device)
    schedule = torch.tensor([8.0, 2.0, 1.0, 0.1, 0.0])
    config = EDMIntegratorConfig()
    plan = EDMRolloutPlan(schedule, (1, 3), device, torch.float32, integrator_config=config)
    assert plan.schedule.device == device
    assert plan.churn_levels.schedule_host.device.type == "cpu"
    torch.testing.assert_close(plan.schedule.cpu(), schedule, atol=0, rtol=0)
    direct = EDMRolloutPlan(schedule.to(device), (1, 3), device, torch.float32, integrator_config=config)
    assert plan.churn_levels.scalar_steps == direct.churn_levels.scalar_steps
    torch.testing.assert_close(plan.churn_levels.sigma_hat, direct.churn_levels.sigma_hat, atol=0, rtol=0)
    with pytest.raises(ValueError, match="nonincreasing"):
        EDMRolloutPlan(torch.tensor([1.0, 2.0]), (1, 3), device, torch.float32)
    if device.type == "cuda":
        with pytest.raises(ValueError, match="device must match"):
            EDMRolloutPlan(schedule.to(device), (1, 3), torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("sigma_hat", "sigma_next"), [(80.0, 2.0), (2.0, 1.0), (0.01, 0.0)])
def test_euler_update_scalar_matches_tensor(device: str, sigma_hat: float, sigma_next: float) -> None:
    from bionemo_ir._torch.sampling.edm import edm_euler_update

    generator = torch.Generator(device=device).manual_seed(0)
    x_noisy = torch.randn(2, 3, 17, 3, device=device, generator=generator) * sigma_hat
    x_denoised = torch.randn(2, 3, 17, 3, device=device, generator=generator)
    fused = edm_euler_update(x_noisy, x_denoised, sigma_hat, sigma_next, 1.5)
    reference = edm_euler_update(
        x_noisy, x_denoised, torch.tensor(sigma_hat, device=device), torch.tensor(sigma_next, device=device), 1.5
    )
    torch.testing.assert_close(fused, reference, atol=1e-5, rtol=1e-5)
