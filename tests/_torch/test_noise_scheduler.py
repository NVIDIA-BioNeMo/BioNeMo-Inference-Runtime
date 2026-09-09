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
"""Guard tests for the shared EDM sampling primitives.

The Boltz / OpenFold3 / Protenix samplers were refactored onto the shared
``_torch/sampling`` runtime (``create_edm_schedule`` +
``centre_random_augmentation`` + ``AF3EDMIntegrator`` driven by
``GenerativeRunner``). Those models have no separate sampler unit tests, so
these tests pin the shared primitives bit-exactly to each model's original
formula / loop (fixed seed), guarding the refactor.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch import sampling as sampling_runtime
from bionemo_ir._torch.layers.random_augmentation import (
    centre_random_augmentation,
    quaternion_to_matrix,
    random_quaternions,
)
from bionemo_ir._torch.modules.openfold3.diffusion_module import OpenFold3DiffusionSampler

_SD, _SMAX, _SMIN, _RHO = 16.0, 160.0, 4e-4, 7.0


def test_graph_safe_sampling_context_preserves_default_rng_stream():
    with torch.random.fork_rng():
        torch.manual_seed(31)
        expected = torch.randn(8)
        expected_next = torch.randn(8)

        torch.manual_seed(31)
        with sampling_runtime.SamplingContext.graph_safe(torch.device("cpu")) as context:
            actual = torch.randn(8, generator=context.generator)
        actual_next = torch.randn(8)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_next, expected_next)


def test_seeded_graph_safe_sampling_context_isolates_default_rng():
    with torch.random.fork_rng():
        torch.manual_seed(31)
        default_state = torch.get_rng_state()

        with sampling_runtime.SamplingContext.graph_safe(torch.device("cpu"), seed=7) as context:
            first = torch.randn(8, generator=context.generator)

        assert torch.equal(torch.get_rng_state(), default_state)
        with sampling_runtime.SamplingContext.graph_safe(torch.device("cpu"), seed=7) as context:
            second = torch.randn(8, generator=context.generator)
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("final", ["keep", "zero", "append_zero"])
def test_edm_types_follow_shared_sampling_contracts(final):
    """Every terminal policy still represents the requested model calls."""
    config = sampling_runtime.EDMScheduleConfig(sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO, final=final)
    assert isinstance(config, sampling_runtime.SamplingScheduleConfig)
    schedule = config.build(num_steps=4)
    assert schedule.shape == (5,)

    integrator_config = sampling_runtime.EDMIntegratorConfig()
    assert isinstance(integrator_config, sampling_runtime.SamplingIntegratorConfig)
    plan = sampling_runtime.EDMRolloutPlan(
        schedule=schedule, coords_shape=(1, 2, 3), device=torch.device("cpu"), dtype=torch.float32
    )
    assert isinstance(plan, sampling_runtime.SamplingRolloutPlan)
    assert plan.num_steps == 4


def test_flow_matching_plan_supports_modality_specific_schedules():
    backbone = sampling_runtime.FlowMatchingSchedule(
        times=torch.linspace(1e-3, 1.0, 5),
        noise_rate=torch.linspace(1.0, 0.0, 4),
    )
    local_latents = sampling_runtime.FlowMatchingSchedule(
        times=torch.linspace(1e-3, 1.0, 5) ** 2,
        noise_rate=torch.zeros(4),
    )
    plan = sampling_runtime.FlowMatchingRolloutPlan(
        schedule={
            "bb_ca": backbone,
            "local_latents": local_latents,
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert isinstance(plan, sampling_runtime.SamplingRolloutPlan)
    assert plan.num_steps == 4


def test_flow_matching_plan_rejects_step_count_mismatch():
    with pytest.raises(ValueError, match="same number of steps"):
        sampling_runtime.FlowMatchingRolloutPlan(
            schedule={
                "a": sampling_runtime.FlowMatchingSchedule(times=torch.linspace(0.0, 1.0, 3)),
                "b": sampling_runtime.FlowMatchingSchedule(times=torch.linspace(0.0, 1.0, 4)),
            },
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "kwargs", [{"gamma0": -0.1}, {"gamma_min": float("inf")}, {"noise_scale": -1.0}, {"step_scale": 0.0}]
)
def test_edm_integrator_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        sampling_runtime.EDMIntegratorConfig(**kwargs)


def test_schedule_openfold3():
    """OF3 ``create_noise_schedule``: n+1 points, no final zero."""
    n = 20
    t = torch.arange(0, 1 + n, dtype=torch.float32) / n
    exp = _SD * (_SMAX ** (1 / _RHO) + t * (_SMIN ** (1 / _RHO) - _SMAX ** (1 / _RHO))) ** _RHO
    got = sampling_runtime.create_edm_schedule(
        num_points=n + 1, sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO, final="keep"
    )
    torch.testing.assert_close(got, exp)
    assert got.shape == (n + 1,) and got[-1] > 0


def test_schedule_protenix():
    """Protenix ``InferenceNoiseScheduler``: N+1 points, final -> 0."""
    n = 20
    step = torch.arange(n + 1, dtype=torch.float32) / n
    exp = _SD * (_SMAX ** (1 / _RHO) + step * (_SMIN ** (1 / _RHO) - _SMAX ** (1 / _RHO))) ** _RHO
    exp[-1] = 0
    got = sampling_runtime.create_edm_schedule(
        num_points=n + 1, sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO, final="zero"
    )
    torch.testing.assert_close(got, exp)
    assert got[-1] == 0


def test_schedule_boltz():
    """Boltz ``sample_schedule``: M points then an appended 0."""
    m = 20
    steps = torch.arange(m, dtype=torch.float32)
    exp = (_SMAX ** (1 / _RHO) + steps / (m - 1) * (_SMIN ** (1 / _RHO) - _SMAX ** (1 / _RHO))) ** _RHO * _SD
    exp = F.pad(exp, (0, 1), value=0.0)
    got = sampling_runtime.create_edm_schedule(
        num_points=m, sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO, final="append_zero"
    )
    torch.testing.assert_close(got, exp)
    assert got.shape == (m + 1,) and got[-1] == 0


def _oss_centre(xl, atom_mask, scale_trans=1.0):
    """Inline reproduction of the OF3 / Protenix ``centre_random_augmentation``
    (AF3 Alg. 19) with the same RNG call order (quaternion rots, then trans)."""
    n = math.prod(xl.shape[:-2])
    rots = quaternion_to_matrix(random_quaternions(n, dtype=xl.dtype, device=xl.device)).reshape(*xl.shape[:-2], 3, 3)
    trans = scale_trans * torch.randn((*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device)
    mean = (xl * atom_mask[..., None]).sum(-2, keepdim=True) / atom_mask[..., None].sum(-2, keepdim=True).clamp(
        min=1e-7
    )
    out = (xl - mean) @ rots.transpose(-1, -2) + trans[..., None, :]
    return out * atom_mask[..., None]


def test_centre_random_augmentation_matches_oss():
    """Bit-exact vs the original impl (masked), under a fixed seed."""
    xl = torch.randn(2, 3, 10, 3)
    mask = torch.ones(2, 1, 10)
    torch.manual_seed(0)
    exp = _oss_centre(xl, mask)
    torch.manual_seed(0)
    got = centre_random_augmentation(xl, mask=mask)
    torch.testing.assert_close(got, exp)


def test_centre_random_augmentation_is_rigid():
    """Centring + rotation/translation preserves pairwise distances."""
    torch.manual_seed(1)
    xl = torch.randn(1, 2, 12, 3)
    out = centre_random_augmentation(xl, mask=torch.ones(1, 1, 12))

    def _pdist(x):
        d = x[..., :, None, :] - x[..., None, :, :]
        return (d**2).sum(-1)

    torch.testing.assert_close(_pdist(out), _pdist(xl), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("final", ["zero", "keep", "append_zero"])
def test_edm_runner_matches_reference(final):
    """The composed runner/integrator matches an inline AF3 Alg. 18 loop."""
    schedule = sampling_runtime.create_edm_schedule(
        num_points=6, sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO, final=final
    )
    coords_shape = (2, 3, 8, 3)
    mask = torch.ones(2, 1, 8)
    g0, gmin, nscale, sscale = 0.8, 1.0, 1.003, 1.5
    integrator = sampling_runtime.AF3EDMIntegrator(
        sampling_runtime.EDMIntegratorConfig(gamma0=g0, gamma_min=gmin, noise_scale=nscale, step_scale=sscale)
    )
    plan = sampling_runtime.EDMRolloutPlan(schedule, coords_shape, torch.device("cpu"), torch.float32, mask)
    context = sampling_runtime.SamplingContext.create(torch.device("cpu"), seed=123)
    got = (
        sampling_runtime.GenerativeRunner()
        .run(
            integrator,
            plan,
            lambda x_noisy, _sigma_hat: x_noisy * 0.7 - 0.1,
            context,
        )
        .final_state
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    x = schedule[0] * torch.randn(coords_shape, generator=generator)
    for c_last, c_tau in zip(schedule[:-1], schedule[1:], strict=True):
        x = centre_random_augmentation(x, mask=mask, generator=generator)
        gamma = g0 if c_tau > gmin else 0.0
        sh = c_last * (gamma + 1)
        x_noisy = x + nscale * torch.sqrt(sh**2 - c_last**2) * torch.randn(x.shape, generator=generator)
        delta = (x_noisy - (x_noisy * 0.7 - 0.1)) / sh
        x = x_noisy + sscale * (c_tau - sh) * delta
    torch.testing.assert_close(got, x)


def test_edm_step_math_with_injected_noise():
    state = torch.tensor([1.0, -2.0])
    noise = torch.tensor([0.25, -0.5])
    sigma_last = torch.tensor(2.0)
    sigma_hat = torch.tensor(2.5)
    sigma_next = torch.tensor(1.5)
    x_noisy = sampling_runtime.edm_churn(state, sigma_last, sigma_hat, noise, 1.003)
    expected_noisy = state + 1.003 * torch.sqrt(sigma_hat**2 - sigma_last**2) * noise
    torch.testing.assert_close(x_noisy, expected_noisy)

    x_denoised = torch.tensor([0.2, -0.3])
    got = sampling_runtime.edm_euler_update(x_noisy, x_denoised, sigma_hat, sigma_next, 1.5)
    expected = x_noisy + 1.5 * (sigma_next - sigma_hat) * (x_noisy - x_denoised) / sigma_hat
    torch.testing.assert_close(got, expected)


def test_edm_template_orders_hooks_around_prediction():
    events: list[str] = []

    class _Before:
        def before_denoise(self, step, _context):
            assert step.denoised_state is None
            events.append("before")
            return step

    class _After:
        def after_denoise(self, step, _context):
            assert step.denoised_state is not None
            events.append("after")
            return step

    hooks = sampling_runtime.DenoiseHookPipeline([_Before()], [_After()])
    integrator = sampling_runtime.AF3EDMIntegrator(
        sampling_runtime.EDMIntegratorConfig(gamma0=0.0, gamma_min=0.0, noise_scale=0.0, step_scale=1.0),
        hooks,
    )
    assert isinstance(integrator, sampling_runtime.DenoiseIntegratorTemplate)
    plan = sampling_runtime.EDMRolloutPlan(
        schedule=torch.tensor([2.0, 1.0, 0.0]),
        coords_shape=(1, 2, 3),
        device=torch.device("cpu"),
        dtype=torch.float32,
        augment_coordinates=False,
    )

    def predict(noisy_state, _sigma_hat):
        events.append("predict")
        return 0.5 * noisy_state

    sampling_runtime.GenerativeRunner().run(
        integrator,
        plan,
        predict,
        sampling_runtime.SamplingContext.create(torch.device("cpu"), seed=3),
    )
    assert events == ["before", "predict", "after"] * 2


def test_openfold3_sampler_composes_shared_edm_runtime():

    class _Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *, xl_noisy, **_kwargs):
            self.calls += 1
            return 0.7 * xl_noisy

    config = SimpleNamespace(gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.5, use_conditioning=True)
    denoiser = _Denoiser()
    sampler = OpenFold3DiffusionSampler(config, denoiser)
    batch = {
        "atom_mask": torch.ones(1, 1, 4),
        "token_mask": torch.ones(1, 1, 2),
    }
    schedule = sampling_runtime.create_edm_schedule(num_points=4, sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO)
    kwargs = {
        "batch": batch,
        "si_input": torch.zeros(1),
        "si_trunk": torch.zeros(1),
        "zij_trunk": torch.zeros(1),
        "noise_schedule": schedule,
        "no_rollout_samples": 2,
        "attn_metadata": None,
        "seed": 7,
    }

    first = sampler(**kwargs)
    torch.manual_seed(123456)
    second = sampler(**kwargs)
    assert first.shape == (1, 2, 4, 3)
    assert denoiser.calls == 2 * (schedule.numel() - 1)
    torch.testing.assert_close(first, second)

    graph_safe_kwargs = {**kwargs, "seed": None}
    torch.manual_seed(11)
    graph_safe_first = sampler(**graph_safe_kwargs)
    torch.manual_seed(11)
    graph_safe_second = sampler(**graph_safe_kwargs)
    assert denoiser.calls == 4 * (schedule.numel() - 1)
    torch.testing.assert_close(graph_safe_first, graph_safe_second)


def test_openfold3_rollout_uses_schedule_dtype_when_atom_mask_is_bool():
    """AF3EDMIntegrator.initialize() calls torch.randn with plan.dtype."""

    class _Denoiser(nn.Module):
        def forward(self, *, xl_noisy, **_kwargs):
            return 0.7 * xl_noisy

    config = SimpleNamespace(gamma_0=0.8, gamma_min=1.0, noise_scale=1.003, step_scale=1.5, use_conditioning=True)
    sampler = OpenFold3DiffusionSampler(config, _Denoiser())
    schedule = sampling_runtime.create_edm_schedule(num_points=3, sigma_data=_SD, s_max=_SMAX, s_min=_SMIN, rho=_RHO)
    coords = sampler(
        batch={"atom_mask": torch.ones(1, 1, 4, dtype=torch.bool), "token_mask": torch.ones(1, 1, 2)},
        si_input=torch.zeros(1),
        si_trunk=torch.zeros(1),
        zij_trunk=torch.zeros(1),
        noise_schedule=schedule,
        no_rollout_samples=2,
        attn_metadata=None,
        seed=7,
    )
    assert coords.dtype == schedule.dtype
    assert coords.shape == (1, 2, 4, 3)
    assert torch.isfinite(coords).all()
