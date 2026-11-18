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
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.layers.random_augmentation import centre_random_augmentation
from bionemo_ir._torch.sampling import AF3EDMIntegrator, EDMIntegratorConfig, EDMRolloutPlan, GenerativeRunner
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.edm import EDMScheduleConfig
from bionemo_ir._torch.sampling.frame_reuse import (
    FRAME_REUSE_EVALUATED_STEPS,
    FrameReuseEDMIntegrator,
    FrameReuseSchedule,
    frame_reuse_schedule,
    require_forced_reuse,
    wrap_for_frame_reuse,
)
from bionemo_ir.models.boltz2.config import AtomDiffusionConfig as Boltz2AtomDiffusionConfig
from bionemo_ir.models.openfold3.config import EDMSamplingConfig as OpenFold3EDMSamplingConfig
from bionemo_ir.models.protenix.config import EDMSamplingConfig as ProtenixEDMSamplingConfig

_STEPS = 200
_FINALS = ("zero", "keep")  # Protenix ends the schedule at 0; OpenFold3 keeps its last level


def _plan(
    device: str,
    atoms: int = 17,
    steps: int = _STEPS,
    final: str = "zero",
    integrator_config: EDMIntegratorConfig | None = None,
) -> EDMRolloutPlan:
    schedule = EDMScheduleConfig(sigma_data=16.0, s_max=160.0, s_min=4e-4, rho=7.0, final=final).build(
        num_steps=steps, dtype=torch.float32, device=torch.device(device)
    )
    return EDMRolloutPlan(
        schedule,
        (1, 5, atoms, 3),
        schedule.device,
        torch.float32,
        integrator_config=integrator_config or EDMIntegratorConfig(),
    )


def _denoiser(x_noisy: torch.Tensor, sigma_hat: torch.Tensor) -> torch.Tensor:
    # Rotation- and translation-equivariant: shrink toward the centroid.
    return 0.9 * x_noisy + 0.1 * x_noisy.mean(dim=-2, keepdim=True)


def test_schedule_counts_and_steps() -> None:
    schedule = FrameReuseSchedule.for_plan(4, _plan("cpu"))
    assert (schedule.start, schedule.end) == (160, 197)
    reused = [step for step in range(_STEPS) if schedule.reuses(step)]
    assert len(reused) == 28 and _STEPS - len(reused) == 172
    assert not any(schedule.reuses(step) for step in (*range(160), 160, 164, 196, 198, 199))
    assert all(step % 4 != 0 for step in reused) and min(reused) == 161 and max(reused) == 197


@pytest.mark.parametrize("final", _FINALS)
@pytest.mark.parametrize("steps", FRAME_REUSE_EVALUATED_STEPS)
def test_window_is_derived_from_the_plan(steps: int, final: str) -> None:
    plan = _plan("cpu", steps=steps, final=final)
    schedule = FrameReuseSchedule.for_plan(4, plan)
    # The first churn-free step is where the next noise level drops to the churn gate, at 80% of the steps.
    assert schedule.start == steps * 4 // 5 and schedule.end == steps - 3
    assert (
        float(plan.churn_levels.gamma_host[schedule.start - 1]) > 0
        and float(plan.churn_levels.gamma_host[schedule.start]) == 0
    )
    schedule.validate(plan)
    window = schedule.end - schedule.start + 1
    fresh_in_window = -(-window // 4)
    assert sum(schedule.reuses(step) for step in range(steps)) == window - fresh_in_window


@pytest.mark.parametrize("period, start, end", [(1, 160, 197), (4, 200, 199), (4, -1, 197)])
def test_schedule_rejects_degenerate_policies(period: int, start: int, end: int) -> None:
    with pytest.raises(ValueError):
        FrameReuseSchedule(period=period, start=start, end=end)


@pytest.mark.parametrize("period", [2, 3, 5, 6, 8])
def test_unevaluated_periods_are_rejected_up_front(period: int) -> None:
    with pytest.raises(ValueError, match=rf"evaluated periods; got {period}"):
        FrameReuseSchedule.for_plan(period, _plan("cpu"))
    with torch.inference_mode(), pytest.raises(ValueError, match=rf"evaluated periods; got {period}"):
        wrap_for_frame_reuse(AF3EDMIntegrator(EDMIntegratorConfig()), period, _plan("cpu"), _denoiser, training=False)


@pytest.mark.parametrize("steps", [49, 50, 75, 200 + 1, 300])
def test_other_step_counts_are_rejected_up_front(steps: int) -> None:
    plan = _plan("cpu", steps=steps)
    with pytest.raises(ValueError, match=rf"only evaluated with .*100, 150, 200, 400.*; got {steps}"):
        FrameReuseSchedule.for_plan(4, plan)
    with pytest.raises(ValueError, match=rf"got {steps}"):
        FrameReuseSchedule(period=4, start=steps * 4 // 5, end=steps - 3).validate(plan)


def test_reuse_needs_coordinate_augmentation() -> None:
    parent = AF3EDMIntegrator(EDMIntegratorConfig())
    plan = dataclasses.replace(_plan("cpu"), augment_coordinates=False, integrator_config=parent.config)
    with torch.inference_mode():
        assert wrap_for_frame_reuse(parent, "auto", plan, _denoiser, training=False) == (parent, _denoiser)
        with pytest.raises(ValueError, match="augmentation"):
            wrap_for_frame_reuse(parent, 4, plan, _denoiser, training=False)
    reuse = FrameReuseEDMIntegrator(parent, FrameReuseSchedule.for_plan(4, plan))
    with SamplingContext.graph_safe(torch.device("cpu"), 7) as context, pytest.raises(ValueError, match="augmentation"):
        reuse.initialize(plan, context)


def test_plan_levels_must_match_the_integrator_config() -> None:
    plan = _plan("cpu")
    other = AF3EDMIntegrator(EDMIntegratorConfig(gamma0=0.0))
    reuse = FrameReuseEDMIntegrator(other, FrameReuseSchedule.for_plan(4, plan))
    with SamplingContext.graph_safe(torch.device("cpu"), 7) as context, pytest.raises(ValueError, match="config"):
        reuse.initialize(plan, context)


def test_churn_steps_cannot_reuse() -> None:
    plan = _plan("cpu")
    with pytest.raises(ValueError, match="churn"):
        FrameReuseSchedule(period=4, start=10, end=197).validate(plan)


def test_rollout_counts_random_streams_and_mapped_prediction() -> None:
    plan = _plan("cpu")
    base = AF3EDMIntegrator(EDMIntegratorConfig())
    reuse = FrameReuseEDMIntegrator(base, FrameReuseSchedule.for_plan(4, plan))
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


@pytest.mark.parametrize(
    "config_cls, default",
    [(ProtenixEDMSamplingConfig, "auto"), (OpenFold3EDMSamplingConfig, "auto"), (Boltz2AtomDiffusionConfig, 0)],
)
def test_sampling_configs_take_auto_off_or_the_evaluated_period(config_cls: type, default: int | str) -> None:
    assert config_cls().frame_reuse_period == default
    for setting in ("auto", 0, 4):
        assert config_cls(frame_reuse_period=setting).frame_reuse_period == setting
    for period in (-1, 1, 2, 3, 5, 6, 8):
        with pytest.raises(ValueError, match=r"0 \(off\) or one of \(4,\).*; got "):
            config_cls(frame_reuse_period=period)
    # Assignments are checked too, and a rejected one leaves the field as it was.
    config = config_cls(frame_reuse_period=4)
    for value in (3, "Auto"):
        with pytest.raises(ValueError, match="frame_reuse_period"):
            config.frame_reuse_period = value
    assert config.frame_reuse_period == 4


def test_protenix_forced_period_needs_an_evaluated_n_step() -> None:
    with pytest.raises(ValueError, match="only evaluated with .*; got 75"):
        ProtenixEDMSamplingConfig(frame_reuse_period=4, n_step=75)
    assert ProtenixEDMSamplingConfig(frame_reuse_period=4, n_step=100).frame_reuse_period == 4
    assert ProtenixEDMSamplingConfig(n_step=75).frame_reuse_period == "auto"
    assert ProtenixEDMSamplingConfig(frame_reuse_period=0, n_step=75).frame_reuse_period == 0
    # On assignment the check runs from either field, and a rejected value is not kept.
    config = ProtenixEDMSamplingConfig(frame_reuse_period=4)
    with pytest.raises(ValueError, match="only evaluated with .*; got 75"):
        config.n_step = 75
    assert config.n_step == 200
    config.frame_reuse_period = "auto"
    config.n_step = 75
    with pytest.raises(ValueError, match="only evaluated with .*; got 75"):
        config.frame_reuse_period = 4
    assert (config.frame_reuse_period, config.n_step) == ("auto", 75)
    config.set_dtype("bfloat16")  # the propagating setters still assign through validation


def test_protenix_sampler_defaults_to_the_config_period() -> None:
    import inspect

    from bionemo_ir._torch.modules.protenix.diffusion import ProtenixDiffusionSampler

    default = inspect.signature(ProtenixDiffusionSampler).parameters["frame_reuse_period"].default
    assert default == ProtenixEDMSamplingConfig().frame_reuse_period


def test_forced_reuse_is_checked_before_any_model_work() -> None:
    for setting in (0, "auto"):
        require_forced_reuse(setting, 75, blocker="steering")
    require_forced_reuse(4, 200)
    with pytest.raises(ValueError, match=r"only evaluated with \(200, 400\) sampling steps; got 100"):
        require_forced_reuse(4, 100, evaluated_steps=(200, 400))
    with pytest.raises(ValueError, match="steering"):
        require_forced_reuse(4, 200, blocker="steering")


def test_models_reject_a_forced_period_before_the_trunk() -> None:
    from bionemo_ir.models.openfold3.modeling import OpenFold3
    from bionemo_ir.models.protenix.modeling import Protenix

    # The check runs before the trunk, so the stand-ins need nothing the trunk would read.
    openfold3 = SimpleNamespace(diffusion_sampler=SimpleNamespace(frame_reuse_period=4), no_rollout_steps=50)
    protenix = SimpleNamespace(diffusion_sampler=SimpleNamespace(frame_reuse_period=4, n_step=50))
    for forward, model in ((OpenFold3.forward, openfold3), (Protenix.forward, protenix)):
        with pytest.raises(ValueError, match="got 75"):
            forward(model, batch={}, num_sampling_steps=75)
        with pytest.raises(ValueError, match="got 50"):  # None takes the model's default step count
            forward(model, batch={}, num_sampling_steps=None)


def test_wrapper_is_inference_only() -> None:
    plan = _plan("cpu")
    parent = AF3EDMIntegrator(EDMIntegratorConfig())
    with torch.inference_mode():
        integrator, predict = wrap_for_frame_reuse(parent, 4, plan, _denoiser, training=False)
        assert integrator.schedule == FrameReuseSchedule.for_plan(4, plan) and callable(predict)
        assert wrap_for_frame_reuse(parent, "auto", plan, _denoiser, training=True) == (parent, _denoiser)
        with pytest.raises(ValueError, match="inference-only"):
            wrap_for_frame_reuse(parent, 4, plan, _denoiser, training=True)
    with pytest.raises(ValueError, match="inference-only"):
        wrap_for_frame_reuse(parent, 4, plan, _denoiser, training=False)


@pytest.mark.parametrize("steps", FRAME_REUSE_EVALUATED_STEPS)
def test_auto_reuses_only_where_it_was_evaluated(steps: int) -> None:
    plan = _plan("cpu", steps=steps)
    with torch.inference_mode():
        assert frame_reuse_schedule("auto", plan, training=False) == FrameReuseSchedule.for_plan(4, plan)
        assert frame_reuse_schedule("auto", plan, training=False, blocker="steering") is None
        assert frame_reuse_schedule("auto", plan, training=True) is None
        boltz_steps = frame_reuse_schedule("auto", plan, training=False, evaluated_steps=(200, 400))
        assert (boltz_steps is not None) == (steps in (200, 400))
    assert frame_reuse_schedule("auto", plan, training=False) is None  # autograd on


def test_auto_runs_exact_where_a_forced_period_raises() -> None:
    plan = _plan("cpu")
    with torch.inference_mode():
        for steps in (2, 50, 75, 201):
            unevaluated = _plan("cpu", steps=steps)
            assert frame_reuse_schedule("auto", unevaluated, training=False) is None
            with pytest.raises(ValueError, match=rf"; got {steps}$"):
                frame_reuse_schedule(4, unevaluated, training=False)
        with pytest.raises(ValueError, match="steering"):
            frame_reuse_schedule(4, plan, training=False, blocker="steering")
        with pytest.raises(ValueError, match="inference-only"):
            frame_reuse_schedule(4, plan, training=True)
        assert frame_reuse_schedule(0, plan, training=True, blocker="steering") is None


def test_auto_raises_on_a_plan_without_prepared_levels() -> None:
    # Every sampler prepares the levels; a plan without them is a sampler bug, not a reason to run exact.
    plan = dataclasses.replace(_plan("cpu"))
    assert plan.churn_levels is None
    with torch.inference_mode(), pytest.raises(ValueError, match="prepared EDM levels"):
        frame_reuse_schedule("auto", plan, training=False)


@pytest.mark.parametrize("final", _FINALS)
@pytest.mark.parametrize("steps", FRAME_REUSE_EVALUATED_STEPS)
def test_auto_runs_exact_without_a_reuse_window(steps: int, final: str) -> None:
    # With gamma_min = 0 every step but a terminal zero churns, so no window precedes the fresh tail.
    config = EDMIntegratorConfig(gamma_min=0.0)
    plan = _plan("cpu", steps=steps, final=final, integrator_config=config)
    with torch.inference_mode():
        assert frame_reuse_schedule("auto", plan, training=False) is None
        with pytest.raises(ValueError, match="churn-free"):
            wrap_for_frame_reuse(AF3EDMIntegrator(config), 4, plan, _denoiser, training=False)
