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

import dataclasses

import pytest
import torch
from pydantic import ValidationError

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.layers.transformers.local_latents_decoder import get_atom37_mask_table, get_residue_atom_mask
from bionemo_ir._torch.layers.triangle_nodes import TriangleMultiplicationNode
from bionemo_ir._torch.modules.rfproteina import (
    GROUP_KEYS,
    UNASSIGNED_TOKEN_INDEX,
    RFProteinaProductSpaceFlowMatcher,
    RFProteinaProductSpaceSampler,
    RFProteinaSamplingOutput,
    apply_insertion_assignment,
    assign_unindexed_tokens,
    create_fraction_partial_schedules,
    create_modality_schedule,
    create_partial_modality_schedule,
    create_partial_schedules,
    extract_groups,
    get_tsr_safe_schedule,
    materialize_atomized_output,
    materialize_nucleic_acid_output,
    partial_num_steps,
    resolve_partial_times,
    validate_nucleic_acid_contract,
    velocity_flow_step,
    velocity_to_clean,
)
from bionemo_ir._torch.modules.rfproteina import representation_builder as representation_builder_module
from bionemo_ir._torch.modules.rfproteina.nucleic_acid import DNA_RESIDUE_TYPE_RANGE, RNA_RESIDUE_TYPE_RANGE
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import (
    FlowMatchingIntegrator,
    FlowMatchingIntegratorConfig,
    FlowMatchingRolloutPlan,
    FlowMatchingSchedule,
)
from bionemo_ir._torch.sampling.random import normal_draws
from bionemo_ir._torch.sampling.runner import GenerativeRunner, SamplingCancelled
from bionemo_ir.configs import BaseConfig
from bionemo_ir.data.schemas import residue_vocabulary
from bionemo_ir.models.optimize_module_setter import AcceleratedConfig, BackendType

# isort: off
from bionemo_ir._torch.modules.rfproteina.atomized import ATOMIZED_TOKEN_TYPE
from bionemo_ir.models.rfproteina import (
    INVERSE_FOLDING_METHODS,
    RESAMPLED_FORWARD_MARGINAL,
    RFProteina,
    RFProteinaConfig,
    RFProteinaDecoderConfig,
    RFProteinaEncoderConfig,
    RFProteinaPartialSimulationConfig,
    RFProteinaPredictionOutput,
    RFProteinaReferenceConfig,
    RFProteinaRepresentationConfig,
    RFProteinaRepresentations,
    RFProteinaSamplingConfig,
    RECYCLE_SCHEDULES,
    recycle_schedule,
    RFProteinaTrunkConfig,
)

# isort: on
from tests._torch import init_module_weights


def _deterministic_sampling_config(num_steps: int = 4) -> RFProteinaSamplingConfig:
    config = RFProteinaSamplingConfig(num_steps=num_steps, self_condition=True)
    config.backbone.step.sampling_mode = "vf"
    config.backbone.noise_schedule.mode = "ode"
    config.local_latents.step.sampling_mode = "vf"
    config.local_latents.noise_schedule.mode = "ode"
    config.backbone.reference.stochastic_centering_mode = "none"
    return config


def test_rfproteina_tsr_safe_schedules_are_monotonic() -> None:
    for mode in ("power", "log", "uniform"):
        schedule = get_tsr_safe_schedule(mode, 10, 2.0, torch.device("cpu"))
        assert schedule.shape == (11,)
        assert schedule[0] > 0
        torch.testing.assert_close(schedule[-1], torch.tensor(1.0))
        assert torch.all(schedule[1:] > schedule[:-1])


def test_rfproteina_partial_time_contract_and_schedule_coupling() -> None:
    sampling = RFProteinaSamplingConfig(num_steps=100)
    device = torch.device("cpu")
    independent = RFProteinaPartialSimulationConfig(
        partial_t_backbone=0.6,
        partial_t_latent=0.4,
        reference_num_steps=100,
    )
    assert resolve_partial_times(independent, sampling, device) == {
        "bb_ca": 0.6,
        "local_latents": 0.4,
    }
    assert partial_num_steps(independent, sampling.num_steps) == 100

    coupled_config = RFProteinaPartialSimulationConfig(
        partial_t_backbone=0.6,
        couple_partial_t=True,
        reference_num_steps=100,
    )
    coupled = resolve_partial_times(coupled_config, sampling, device)
    reference_backbone = create_modality_schedule(sampling.backbone, 100, device).times
    reference_latent = create_modality_schedule(sampling.local_latents, 100, device).times
    index = int((reference_backbone >= 0.6 - 1e-6).nonzero(as_tuple=True)[0][0])
    assert coupled["bb_ca"] == reference_backbone[index].item()
    assert coupled["local_latents"] == reference_latent[index].item()
    assert coupled["local_latents"] != pytest.approx(0.6)

    fraction_config = RFProteinaPartialSimulationConfig(
        partial_trajectory_fraction_steps=0.3,
        reference_num_steps=100,
    )
    fraction_times = resolve_partial_times(fraction_config, sampling, device)
    assert fraction_times["bb_ca"] == reference_backbone[70].item()
    assert fraction_times["local_latents"] == reference_latent[70].item()
    assert partial_num_steps(fraction_config, sampling.num_steps) == 30


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"partial_t_backbone": 0.5}, "requires couple_partial_t"),
        ({"partial_t_backbone": 0.5, "partial_t_latent": 0.5, "couple_partial_t": True}, "requires exactly one"),
        (
            {"partial_t_backbone": 0.5, "partial_t_latent": 0.5, "partial_trajectory_fraction_steps": 0.2},
            "mutually exclusive",
        ),
        ({"partial_trajectory_fraction_steps": 0.0}, "must be in"),
        ({"partial_t_latent": 1.0, "couple_partial_t": True}, "must be in"),
        (
            {"inverse_folding": True, "partial_t_backbone": 0.2},
            "mutually exclusive",
        ),
        (
            {"inverse_folding": True, "inverse_folding_method": "m3"},
            "inverse_folding_method",
        ),
    ],
)
def test_rfproteina_partial_time_contract_rejects_invalid_combinations(kwargs, match) -> None:
    with pytest.raises(ValidationError, match=match):
        RFProteinaPartialSimulationConfig(**kwargs)


def test_rfproteina_inverse_folding_resolves_to_full_noise() -> None:
    sampling = RFProteinaSamplingConfig(num_steps=100)
    device = torch.device("cpu")
    config = RFProteinaPartialSimulationConfig(inverse_folding=True)
    assert config.inverse_folding_method in INVERSE_FOLDING_METHODS
    assert config.inverse_folding_method == RESAMPLED_FORWARD_MARGINAL
    assert resolve_partial_times(config, sampling, device) == {"bb_ca": 0.0, "local_latents": 0.0}
    assert partial_num_steps(config, sampling.num_steps) == 100


@pytest.mark.parametrize("frac", [0.05, 0.10, 0.20, 0.50, 1.00])
def test_rfproteina_partial_schedule_starts_at_the_noised_time(frac: float) -> None:
    sampling = RFProteinaSamplingConfig(num_steps=100)
    device = torch.device("cpu")
    config = RFProteinaPartialSimulationConfig(
        partial_trajectory_fraction_steps=frac,
        reference_num_steps=100,
    )
    start_times = resolve_partial_times(config, sampling, device)
    num_steps = partial_num_steps(config, sampling.num_steps)
    schedules = create_partial_schedules(sampling, start_times, num_steps, device)
    for modality, start in start_times.items():
        assert schedules[modality].times[0].item() == pytest.approx(start, abs=1e-6)
        assert schedules[modality].times[-1].item() == pytest.approx(1.0, abs=1e-6)
    if frac == 0.05:
        # Regression for the removed 1-1/nsteps clamp: five steps used to start
        # the schedule at 0.80 while the state was noised to ~0.997.
        assert start_times["bb_ca"] > 0.9
        assert start_times["local_latents"] > 0.9


def test_rfproteina_partial_fraction_that_rounds_to_zero_steps_raises() -> None:
    sampling = RFProteinaSamplingConfig(num_steps=2)
    device = torch.device("cpu")
    config = RFProteinaPartialSimulationConfig(partial_trajectory_fraction_steps=0.05)
    with pytest.raises(ValueError, match="zero sampling steps"):
        resolve_partial_times(config, sampling, device)


def test_rfproteina_partial_schedule_full_tail_regression() -> None:
    sampling = RFProteinaSamplingConfig(num_steps=12)
    device = torch.device("cpu")
    full = {
        "bb_ca": create_modality_schedule(sampling.backbone, 12, device),
        "local_latents": create_modality_schedule(sampling.local_latents, 12, device),
    }
    start_times = {modality: schedule.times[0].item() for modality, schedule in full.items()}
    partial = create_partial_schedules(sampling, start_times, 12, device)
    for modality in full:
        torch.testing.assert_close(partial[modality].times, full[modality].times)
        torch.testing.assert_close(partial[modality].noise_rate, full[modality].noise_rate)

    shortened = create_partial_modality_schedule(sampling.backbone, 0.8, 12, device)
    assert shortened.num_steps == 12
    assert shortened.times[0] >= 0.8 - 1e-2
    assert shortened.times[-1] == 1.0


def _densified_sampling(num_steps: int, tail_steps: int = 10) -> RFProteinaSamplingConfig:
    """The released sampler's shape: the last ``tail_steps`` intervals of each modality split in two."""
    sampling = RFProteinaSamplingConfig(num_steps=num_steps)
    for modality in (sampling.backbone, sampling.local_latents):
        modality.schedule.tail_steps = tail_steps
        modality.schedule.tail_step_multiplier = 2
    return sampling


@pytest.mark.parametrize("num_steps", [60, 61, 100])
@pytest.mark.parametrize("fraction", [1 / 60, 1 / 3, 0.375, 0.5, 1.0])
def test_rfproteina_fraction_corruption_uses_the_configured_schedule(num_steps: int, fraction: float) -> None:
    sampling = _densified_sampling(num_steps)
    device = torch.device("cpu")
    config = RFProteinaPartialSimulationConfig(partial_trajectory_fraction_steps=fraction)

    start_times = resolve_partial_times(config, sampling, device)

    start_step = num_steps - round(fraction * num_steps)
    for modality, modality_config in (("bb_ca", sampling.backbone), ("local_latents", sampling.local_latents)):
        full = create_modality_schedule(modality_config, num_steps, device)
        assert start_times[modality] == full.times[start_step].item()


@pytest.mark.parametrize("tail_steps", [0, 10])
@pytest.mark.parametrize("scaled_drift", [False, True])
@pytest.mark.parametrize("remaining", [1, 2, 20, 21, 28, 60])
def test_rfproteina_fraction_integrates_the_tail_of_the_full_schedule(
    tail_steps: int, scaled_drift: bool, remaining: int
) -> None:
    sampling = _densified_sampling(60, tail_steps)
    for modality_config in (sampling.backbone, sampling.local_latents):
        modality_config.step.sampling_mode = "vf_tsr"
        modality_config.step.tsr_scaled_score_drift = scaled_drift
    device = torch.device("cpu")
    config = RFProteinaPartialSimulationConfig(partial_trajectory_fraction_steps=remaining / 60)
    assert partial_num_steps(config, 60) == remaining

    schedules = create_fraction_partial_schedules(sampling, remaining, device)
    start_times = resolve_partial_times(config, sampling, device)

    for modality, modality_config in (("bb_ca", sampling.backbone), ("local_latents", sampling.local_latents)):
        full = create_modality_schedule(modality_config, 60, device)
        torch.testing.assert_close(schedules[modality].times, full.times[-remaining - 1 :], rtol=0, atol=0)
        torch.testing.assert_close(schedules[modality].noise_rate, full.noise_rate[-remaining:], rtol=0, atol=0)
        assert schedules[modality].times[0].item() == start_times[modality]
        assert schedules[modality].times[-1].item() == 1.0
        assert (schedules[modality].times.diff() > 0).all()


@pytest.mark.parametrize("remaining", [0, 61])
def test_rfproteina_fraction_schedule_rejects_an_impossible_step_count(remaining: int) -> None:
    with pytest.raises(ValueError, match="num_steps must be in"):
        create_fraction_partial_schedules(_densified_sampling(60), remaining, torch.device("cpu"))


def test_rfproteina_velocity_to_clean() -> None:
    state = torch.tensor([[[1.0, 2.0]]])
    velocity = torch.tensor([[[2.0, -2.0]]])
    time = torch.tensor([0.25])
    mask = torch.tensor([[True]])
    clean = velocity_to_clean(state, velocity, time, mask)
    torch.testing.assert_close(clean, torch.tensor([[[2.5, 0.5]]]))


def test_rfproteina_product_sampler_self_conditioning() -> None:
    config = _deterministic_sampling_config()
    sampler = RFProteinaProductSpaceSampler(config, latent_dim=4)
    mask = torch.tensor([[True, True, False]], device="cuda")
    initial = {
        "bb_ca": torch.randn(1, 3, 3, device="cuda"),
        "local_latents": torch.randn(1, 3, 4, device="cuda"),
    }
    initial = {key: value * mask.unsqueeze(-1) for key, value in initial.items()}
    self_condition_flags: list[bool] = []

    def predict(state, times, self_conditioning, active, recycle_depth):
        del times, self_conditioning, recycle_depth
        self_condition_flags.append(active)
        return {
            "bb_ca": {"v": torch.zeros_like(state["bb_ca"])},
            "local_latents": {"v": torch.zeros_like(state["local_latents"])},
        }

    output = sampler.sample(
        predict,
        mask,
        torch.float32,
        initial_state=initial,
        save_trajectory_every=2,
    )
    torch.testing.assert_close(output.state["bb_ca"], initial["bb_ca"])
    torch.testing.assert_close(output.state["local_latents"], initial["local_latents"])
    assert self_condition_flags == [False, True, True, True]
    assert len(output.trajectory) == 3


def test_rfproteina_product_sampler_applies_recycle_schedule() -> None:
    config = RFProteinaSamplingConfig(
        num_steps=4,
        n_recycle_schedule=[(1, 1), (1, 2), (2, 3)],
    )
    assert config.recycle_depths() == (1, 2, 3, 3)
    assert config.recycle_depths(2) == (3, 3)

    sampler = RFProteinaProductSpaceSampler(config, latent_dim=4)
    mask = torch.ones(1, 2, device="cuda", dtype=torch.bool)
    initial = {
        "bb_ca": torch.zeros(1, 2, 3, device="cuda"),
        "local_latents": torch.zeros(1, 2, 4, device="cuda"),
    }
    seen_depths: list[int] = []

    def predict(state, times, self_conditioning, active, recycle_depth):
        del times, self_conditioning, active
        seen_depths.append(recycle_depth)
        return {modality: {"v": torch.zeros_like(value)} for modality, value in state.items()}

    sampler.sample(predict, mask, torch.float32, initial_state=initial, seed=11)
    assert seen_depths == [1, 2, 3, 3]


def test_rfproteina_inverse_folding_resamples_backbone_and_restores_target() -> None:
    config = _deterministic_sampling_config(num_steps=3)
    sampler = RFProteinaProductSpaceSampler(config, latent_dim=4)
    mask = torch.tensor([[True, True, False]], device="cuda")
    target = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]], device="cuda")
    clean = {
        "bb_ca": target,
        "local_latents": torch.zeros(1, 3, 4, device="cuda"),
    }
    seen_self_cond: list[torch.Tensor] = []
    seen_state_backbone: list[torch.Tensor] = []

    def predict(state, times, self_conditioning, active, recycle_depth):
        del times, active, recycle_depth
        seen_self_cond.append(self_conditioning["bb_ca"].detach().clone())
        seen_state_backbone.append(state["bb_ca"].detach().clone())
        return {
            "bb_ca": {"v": torch.full_like(state["bb_ca"], 99.0)},
            "local_latents": {"v": torch.ones_like(state["local_latents"])},
        }

    output = sampler.sample_partial(
        predict,
        clean,
        {"bb_ca": 0.0, "local_latents": 0.0},
        num_steps=3,
        padding_mask=mask,
        dtype=torch.float32,
        resample_target_backbone=True,
        target_backbone=target,
        seed=7,
    )
    torch.testing.assert_close(output.sampling.state["bb_ca"], target)
    assert not torch.equal(output.sampling.state["local_latents"], clean["local_latents"])
    assert len(seen_state_backbone) == 3
    # First step is a forward-marginal draw, not the model's previous backbone update.
    assert not torch.allclose(seen_state_backbone[0], torch.full_like(target, 99.0))
    torch.testing.assert_close(seen_self_cond[1], target)
    torch.testing.assert_close(seen_self_cond[2], target)


def test_rfproteina_product_sampler_runs_partial_schedule_tail() -> None:
    config = _deterministic_sampling_config(num_steps=8)
    sampler = RFProteinaProductSpaceSampler(config, latent_dim=4)
    mask = torch.tensor([[True, True, True, False]], device="cuda")
    clean = {
        "bb_ca": torch.randn(1, 4, 3, device="cuda") * mask[..., None],
        "local_latents": torch.randn(1, 4, 4, device="cuda") * mask[..., None],
    }
    flags = []

    def predict(state, times, self_conditioning, active, recycle_depth):
        del times, self_conditioning, recycle_depth
        flags.append(active)
        return {modality: {"v": torch.zeros_like(value)} for modality, value in state.items()}

    output = sampler.sample_partial(
        predict,
        clean,
        {"bb_ca": 0.6, "local_latents": 0.4},
        num_steps=3,
        padding_mask=mask,
        dtype=torch.float32,
        seed=71,
        save_trajectory_every=2,
    )

    assert output.num_steps == 3
    assert output.sampling.seed == 71
    assert flags == [False, True, True]
    assert len(output.sampling.trajectory) == 2
    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(
            output.sampling.state[modality],
            output.corruption.corrupted_state[modality],
        )


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


@pytest.mark.parametrize("sampling_mode", ["vf", "vf_tsr"])
def test_rfproteina_sampler_matches_hand_rolled_loop(sampling_mode: str) -> None:
    """Driving the shared runner must reproduce RFProteína's original loop.

    The reference below is the loop the sampler used before it moved onto
    ``GenerativeRunner``. ``vf_tsr`` with a stochastic noise schedule also
    pins the order of the random draws, which the refactor could otherwise
    have permuted between modalities.
    """
    config = RFProteinaSamplingConfig(num_steps=5)
    for modality in (config.backbone, config.local_latents):
        modality.step.sampling_mode = sampling_mode
        modality.noise_schedule.mode = "ode" if sampling_mode == "vf" else "1/t"

    latent_dim = 4
    mask = torch.tensor([[True] * 6 + [False] * 2], device="cuda")
    can_be_encoded = torch.tensor([[True] * 5 + [False] * 3], device="cuda")
    torch.manual_seed(7)
    velocities = [
        {
            "bb_ca": torch.randn(1, 8, 3, device="cuda"),
            "local_latents": torch.randn(1, 8, latent_dim, device="cuda"),
        }
        for _ in range(config.num_steps)
    ]

    def predict(state, times, self_conditioning, active, recycle_depth):
        del state, times, self_conditioning, active, recycle_depth
        step = predict.calls
        predict.calls += 1
        return {
            "bb_ca": {"v": velocities[step]["bb_ca"]},
            "local_latents": {"v": velocities[step]["local_latents"]},
        }

    predict.calls = 0

    sampler = RFProteinaProductSpaceSampler(config, latent_dim=latent_dim)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234)
    output = sampler.sample(
        predict, mask, torch.float32, can_be_encoded=can_be_encoded, generator=generator, save_trajectory_every=2
    )

    # Reference: the pre-refactor loop, driven by an identical random stream.
    from bionemo_ir._torch.modules.rfproteina.sampler import sample_reference_state

    reference_generator = torch.Generator(device="cuda")
    reference_generator.manual_seed(1234)
    state = sample_reference_state(
        mask, latent_dim, torch.float32, config, generator=reference_generator, can_be_encoded=can_be_encoded
    )
    schedules = {
        "bb_ca": create_modality_schedule(config.backbone, config.num_steps, mask.device),
        "local_latents": create_modality_schedule(config.local_latents, config.num_steps, mask.device),
    }
    modality_configs = {"bb_ca": config.backbone, "local_latents": config.local_latents}
    clean = {key: torch.zeros_like(value) for key, value in state.items()}
    trajectory_length = 0
    for step_index in range(config.num_steps):
        times = {modality: schedules[modality].times[step_index].expand(1) for modality in ("bb_ca", "local_latents")}
        next_state = {}
        for modality in ("bb_ca", "local_latents"):
            velocity = velocities[step_index][modality]
            clean[modality] = velocity_to_clean(state[modality], velocity, times[modality], mask)
            schedule = schedules[modality]
            next_state[modality] = velocity_flow_step(
                state[modality],
                velocity,
                times[modality],
                schedule.times[step_index + 1] - schedule.times[step_index],
                schedule.noise_rate[step_index],
                mask,
                modality_configs[modality].step,
                generator=reference_generator,
            )
        next_state["local_latents"] = next_state["local_latents"] * can_be_encoded.unsqueeze(-1).to(torch.float32)
        state = next_state
        if step_index % 2 == 0 or step_index == config.num_steps - 1:
            trajectory_length += 1

    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(output.state[modality], state[modality])
        torch.testing.assert_close(output.clean_prediction[modality], clean[modality])
    assert len(output.trajectory) == trajectory_length
    assert output.seed == 1234
    assert predict.calls == config.num_steps


def test_rfproteina_reference_state_zeroes_non_encodable_latents() -> None:
    """Upstream masks the latent noise before the first velocity call.

    ``product_space_flow_matcher.py`` zeroes non-encodable local latents
    immediately after drawing the reference noise. Masking only at the end of
    each step would let atomized tokens reach the trunk with noise on step 0
    and diverge for the rest of the trajectory.
    """
    from bionemo_ir._torch.modules.rfproteina.sampler import sample_reference_state

    config = RFProteinaSamplingConfig(num_steps=3)
    padding_mask = torch.tensor([[True] * 6 + [False] * 2], device="cuda")
    can_be_encoded = torch.tensor([[True] * 4 + [False] * 4], device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(11)

    state = sample_reference_state(
        padding_mask, 4, torch.float32, config, generator=generator, can_be_encoded=can_be_encoded
    )

    latents = state["local_latents"]
    assert (latents[:, 4:] == 0).all()
    assert (latents[:, :4] != 0).any()
    # Padding must still zero the backbone, which carries no latent mask.
    assert (state["bb_ca"][:, 6:] == 0).all()


def test_rfproteina_reference_state_masking_does_not_move_the_stream() -> None:
    """Masking happens after the draw, so the RNG stream is unaffected."""
    from bionemo_ir._torch.modules.rfproteina.sampler import sample_reference_state

    config = RFProteinaSamplingConfig(num_steps=3)
    padding_mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    can_be_encoded = torch.tensor([[True] * 4 + [False] * 4], device="cuda")

    states = []
    for mask in (None, can_be_encoded):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(11)
        sample_reference_state(padding_mask, 4, torch.float32, config, generator=generator, can_be_encoded=mask)
        states.append(torch.randn(4, device="cuda", generator=generator))

    torch.testing.assert_close(states[0], states[1])


def test_rfproteina_partial_corruption_matches_upstream_rng_order() -> None:
    config = RFProteinaSamplingConfig(num_steps=3)
    matcher = RFProteinaProductSpaceFlowMatcher(config, latent_dim=4)
    padding_mask = torch.tensor([[True, True, True, False]], device="cuda")
    can_be_encoded = torch.tensor([[True, True, False, False]], device="cuda")
    clean = {
        "bb_ca": torch.randn(1, 4, 3, device="cuda") * padding_mask[..., None],
        "local_latents": torch.randn(1, 4, 4, device="cuda") * can_be_encoded[..., None],
    }
    times = {"bb_ca": 0.25, "local_latents": 0.75}
    generator = torch.Generator(device="cuda").manual_seed(123)
    output = matcher.corrupt_state(
        clean,
        times,
        padding_mask,
        generator=generator,
        can_be_encoded=can_be_encoded,
    )

    # Upstream corrupt_batch draws both raw modality noises before applying
    # the backbone's interpolation-time x_0 centering offset.
    reference_generator = torch.Generator(device="cuda").manual_seed(123)
    backbone_noise = torch.randn(1, 4, 3, device="cuda", generator=reference_generator)
    backbone_noise *= padding_mask[..., None]
    count = padding_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
    backbone_noise = (backbone_noise - backbone_noise.sum(dim=-2, keepdim=True) / count) * padding_mask[..., None]
    latent_noise = torch.randn(1, 4, 4, device="cuda", generator=reference_generator)
    latent_noise *= can_be_encoded[..., None]
    offset = torch.randn(1, 3, device="cuda", generator=reference_generator)
    centered_backbone = backbone_noise + config.backbone.reference.stochastic_centering_scale * offset.unsqueeze(
        -2
    ) * padding_mask.unsqueeze(-1)
    expected_backbone = 0.75 * centered_backbone + 0.25 * clean["bb_ca"]
    expected_latents = 0.25 * latent_noise + 0.75 * clean["local_latents"]

    torch.testing.assert_close(output.reference_state["bb_ca"], backbone_noise)
    torch.testing.assert_close(output.reference_state["local_latents"], latent_noise)
    torch.testing.assert_close(output.corrupted_state["bb_ca"], expected_backbone)
    torch.testing.assert_close(output.corrupted_state["local_latents"], expected_latents)
    torch.testing.assert_close(
        torch.randn(5, device="cuda", generator=generator),
        torch.randn(5, device="cuda", generator=reference_generator),
    )


def _centering_config(scope: str, group_keys: tuple[str, ...] = ()) -> RFProteinaSamplingConfig:
    config = RFProteinaSamplingConfig(num_steps=3)
    reference = config.backbone.reference
    reference.stochastic_centering_scope = scope
    reference.stochastic_centering_group_keys = group_keys
    return config


def _draw_backbone(config: RFProteinaSamplingConfig, padding_mask, **kwargs):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(5)
    matcher = RFProteinaProductSpaceFlowMatcher(config, latent_dim=4)
    return matcher.sample_noise(padding_mask, generator=generator, **kwargs)["bb_ca"]


def test_rfproteina_reference_scopes_differ_on_conditioned_tokens() -> None:
    """Only the ``all`` scope translates coordinate-conditioned tokens.

    Conditioned tokens are pinned to absolute coordinates, so moving them would
    break the very conditioning they carry. ``all`` is the exception that keeps
    the whole system rigid, which leaves those tokens where they started
    relative to the rest.
    """
    padding_mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    coordinate_condition_mask = torch.tensor([[True] * 4 + [False] * 4], device="cuda")

    pinned = _draw_backbone(
        _centering_config("without_absolute_coords"), padding_mask, coordinate_condition_mask=coordinate_condition_mask
    )
    moved = _draw_backbone(_centering_config("all"), padding_mask, coordinate_condition_mask=coordinate_condition_mask)
    plain = _draw_backbone(_centering_config("without_absolute_coords"), padding_mask)

    # Offsetting everything is the same as having nothing to pin.
    torch.testing.assert_close(moved, plain)
    # The unconditioned tail sees the same offset either way; the conditioned
    # head only moves under ``all``.
    torch.testing.assert_close(pinned[:, 4:], moved[:, 4:])
    assert not torch.allclose(pinned[:, :4], moved[:, :4])


def test_rfproteina_per_group_centering_moves_groups_independently() -> None:
    """Each group gets its own translation, so chains move independently.

    Without this, every chain in a complex would share one offset and the model
    would never see two chains placed independently of one another.
    """
    padding_mask = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    groups = {
        "chain_iid": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device="cuda"),
    }
    grouped = _draw_backbone(_centering_config("per_group", ("chain_iid",)), padding_mask, groups=groups)
    shared = _draw_backbone(_centering_config("without_absolute_coords"), padding_mask)

    # Both draws consume the base noise and then one offset, so the first chain
    # lands exactly on the single-offset result and only the second diverges.
    torch.testing.assert_close(grouped[:, :4], shared[:, :4])
    difference = grouped[:, 4:] - shared[:, 4:]
    assert difference.abs().max() > 0
    # A translation, so every token in the chain shifts by the same vector.
    torch.testing.assert_close(difference, difference[:, :1].expand_as(difference))


def test_rfproteina_per_group_centering_requires_its_group_tensors() -> None:
    """A missing group tensor must name itself rather than silently no-op."""
    config = _centering_config("per_group", ("chain_iid",))
    padding_mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")

    with pytest.raises(ValueError, match="chain_iid"):
        _draw_backbone(config, padding_mask, groups={"molecule_iid": None})


def test_rfproteina_per_group_scope_rejects_empty_group_keys() -> None:
    """Per-group centering without keys would silently degrade to one group."""
    with pytest.raises(ValidationError, match="per_group"):
        RFProteinaReferenceConfig(stochastic_centering_scope="per_group")


def test_rfproteina_reference_config_rejects_unknown_group_keys() -> None:
    with pytest.raises(ValidationError, match="entity_iid"):
        RFProteinaReferenceConfig(
            stochastic_centering_scope="per_group", stochastic_centering_group_keys=("entity_iid",)
        )


def test_rfproteina_extract_groups_reports_absent_tensors_as_none() -> None:
    """Absent keys stay in the mapping so a scope can report them by name."""
    coordinate_group = torch.full((1, 4, 37), -1, device="cuda")
    coordinate_group[:, :2, 3] = 2
    groups = extract_groups(
        {
            "condition_CoordinateGroup_1_atom": coordinate_group,
            "molecule_iid": torch.zeros(1, 4, dtype=torch.long, device="cuda"),
        }
    )

    assert set(groups) == set(GROUP_KEYS)
    assert groups["chain_iid"] is None
    # A token takes the highest coordinate group any of its atoms carries.
    torch.testing.assert_close(groups["coord_group"], torch.tensor([[2, 2, -1, -1]], device="cuda"))


def test_rfproteina_sampler_seed_is_reproducible_and_reported() -> None:
    """An explicit seed must pin the rollout and come back on the output."""
    config = RFProteinaSamplingConfig(num_steps=3)
    sampler = RFProteinaProductSpaceSampler(config, latent_dim=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")

    def predict(state, times, self_conditioning, active, recycle_depth):
        del times, self_conditioning, active, recycle_depth
        return {modality: {"v": torch.zeros_like(value)} for modality, value in state.items()}

    first = sampler.sample(predict, mask, torch.float32, seed=99)
    second = sampler.sample(predict, mask, torch.float32, seed=99)
    other = sampler.sample(predict, mask, torch.float32, seed=100)

    assert first.seed == 99
    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(first.state[modality], second.state[modality])
        assert not torch.allclose(first.state[modality], other.state[modality])


def _small_trunk_config() -> RFProteinaTrunkConfig:
    return RFProteinaTrunkConfig(
        token_dim=32,
        pair_dim=16,
        condition_dim=8,
        latent_dim=4,
        num_layers=2,
        num_heads=4,
        trimul_hidden_dim=8,
        pair_write="sum",
        recycle_enabled=False,
        dtype="float32",
        pairwise_attention_backend="SDPA",
    )


def _small_decoder_config(**updates) -> RFProteinaDecoderConfig:
    values = {
        "token_dim": 32,
        "pair_dim": 16,
        "condition_dim": 8,
        "latent_dim": 4,
        "num_layers": 2,
        "num_heads": 4,
        "input_schema": "legacy",
        "num_residue_types": 20,
        "sequence_separation_bins": 127,
        "use_pair_bias": True,
        "dtype": "float32",
        "pairwise_attention_backend": "SDPA",
    }
    values.update(updates)
    return RFProteinaDecoderConfig(**values)


def _small_encoder_config(**updates) -> RFProteinaEncoderConfig:
    values = {
        "token_dim": 32,
        "pair_dim": 16,
        "condition_dim": 8,
        "latent_dim": 4,
        "num_layers": 2,
        "num_heads": 4,
        "input_schema": "legacy",
        "num_residue_types": 20,
        "sequence_separation_bins": 127,
        "use_pair_bias": True,
        "dtype": "float32",
        "transformer_dtype": "float32",
        "pairwise_attention_backend": "SDPA",
        "precompute_bias": True,
    }
    values.update(updates)
    return RFProteinaEncoderConfig(**values)


def _native_static_features(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, tokens = mask.shape
    device = mask.device
    return {
        "padding_mask": mask,
        "token_type": torch.zeros(batch, tokens, dtype=torch.long, device=device),
        "atom37_mask": mask.unsqueeze(-1).expand(-1, -1, 37),
        "condition_CoordinateGroup_1_atom": torch.full((batch, tokens, 37), -1, dtype=torch.long, device=device),
        "condition_cath-label_1_residue": torch.full((batch, tokens, 3), -1, dtype=torch.long, device=device),
        "condition_index-group_1_token": torch.zeros(batch, tokens, dtype=torch.long, device=device),
        "molecule_iid": torch.zeros(batch, tokens, dtype=torch.long, device=device),
        "chain_iid": torch.zeros(batch, tokens, dtype=torch.long, device=device),
        "token_bond_path_length": torch.zeros(batch, tokens, tokens, dtype=torch.long, device=device),
    }


def _atomized_static_features() -> dict[str, torch.Tensor]:
    """Build a tiny protein-plus-ligand tensor batch."""
    mask = torch.ones(1, 5, dtype=torch.bool)
    static_features = _native_static_features(mask)
    static_features["token_type"] = torch.tensor([[0, 0, 3, 3, 4]])
    atom37_mask = torch.zeros(1, 5, 37, dtype=torch.bool)
    atom37_mask[:, :2, :4] = True
    atom37_mask[:, 2:4, 1] = True
    static_features["atom37_mask"] = atom37_mask
    static_features["residue_type"] = torch.tensor([[1, 2, 32, 32, 0]])
    static_features["can_be_encoded"] = torch.tensor([[True, True, False, False, False]])

    static_features["element"] = torch.zeros(1, 5, 37, dtype=torch.long)
    static_features["element"][0, 2:4, 1] = torch.tensor([6, 8])
    static_features["charge"] = torch.zeros(1, 5, 37, dtype=torch.long)
    static_features["charge"][0, 3, 1] = -1
    static_features["chirality"] = torch.zeros(1, 5, 18)
    static_features["molecule_iid"] = torch.tensor([[0, 0, 1, 1, -1]])
    static_features["chain_iid"] = torch.tensor([[0, 0, 1, 1, -1]])

    bond_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    bond_mask[0, 2, 3] = bond_mask[0, 3, 2] = True
    static_features["bond_mask"] = bond_mask
    static_features["bond_order"] = bond_mask.to(torch.long)
    static_features["is_aromatic"] = torch.zeros_like(bond_mask)
    static_features["token_bond_path_length"] = torch.zeros(1, 5, 5, dtype=torch.long)

    coordinate_mask = torch.zeros(1, 5, 37, dtype=torch.bool)
    coordinate_mask[0, 2, 1] = True
    static_features["mask_coordinate_1_atom"] = coordinate_mask
    coordinates = torch.zeros(1, 5, 37, 3)
    coordinates[0, 2, 1] = torch.tensor([40.0, 50.0, 60.0])
    static_features["condition_coordinate_1_atom"] = coordinates
    return static_features


def test_rfproteina_materializes_atomized_product_space_output() -> None:
    static_features = _atomized_static_features()
    bb_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )

    atomized = materialize_atomized_output(bb_ca, static_features)

    assert atomized is not None
    torch.testing.assert_close(
        atomized.token_mask,
        torch.tensor([[False, False, True, True, False]]),
    )
    torch.testing.assert_close(atomized.generated_coordinates[0, 2, 1], bb_ca[0, 2])
    torch.testing.assert_close(atomized.generated_coordinates[0, 3, 1], bb_ca[0, 3])
    torch.testing.assert_close(atomized.coordinates[0, 2, 1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(atomized.coordinates[0, 3, 1], bb_ca[0, 3])
    assert not torch.any(atomized.coordinates[0, 2:4, 2:])
    assert atomized.fixed_token_mask[0, 2]
    assert not atomized.fixed_token_mask[0, 3]
    torch.testing.assert_close(atomized.residue_types[0, 2:4], torch.tensor([32, 32]))
    torch.testing.assert_close(atomized.element[0, 2:4, 1], torch.tensor([6, 8]))
    assert atomized.charge[0, 3, 1] == -1
    assert atomized.bond_mask[0, 2, 3]
    assert atomized.bond_order[0, 2, 3] == 1

    static_features["residue_type"][0, 2] = 31
    with pytest.raises(ValueError, match="unified residue_type 32"):
        materialize_atomized_output(bb_ca, static_features)


def test_rfproteina_prediction_exposes_ligand_safe_complex_view() -> None:
    static_features = _atomized_static_features()
    # The model must enforce this invariant from token_type even if a caller
    # accidentally marks ligand atoms as autoencoder-compatible.
    static_features["can_be_encoded"][0, 2:4] = True
    bb_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    state = {
        "bb_ca": bb_ca,
        "local_latents": torch.zeros(1, 5, 4),
    }

    class StaticSampler(torch.nn.Module):
        group_keys = frozenset()

        def __init__(self) -> None:
            super().__init__()
            self.can_be_encoded = None

        def sample(self, **kwargs):
            self.can_be_encoded = kwargs["can_be_encoded"]
            return RFProteinaSamplingOutput(
                state=state,
                clean_prediction=state,
                insertion_logits=None,
                trajectory=[],
                seed=kwargs["seed"],
            )

    class RecordingDecoder(torch.nn.Module):
        num_residue_types = 33
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.token_type = None

        def forward(self, backbone_ca, local_latents, mask, token_type=None, **kwargs):
            del backbone_ca, local_latents, kwargs
            self.token_type = token_type
            coordinates = torch.full((*mask.shape, 37, 3), 9.0)
            logits = torch.zeros(*mask.shape, self.num_residue_types)
            residue_types = torch.where(token_type == 3, 32, 1) * mask
            return {
                "coords": coordinates,
                "sequence_logits": logits,
                "residue_type": residue_types,
            }

    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    sampler = StaticSampler()
    model.sampler = sampler
    decoder = RecordingDecoder()
    model.decoder = decoder

    output = model(static_features, seed=23)

    torch.testing.assert_close(decoder.token_type, torch.tensor([[0, 0]]))
    assert not torch.any(sampler.can_be_encoded[0, 2:4])
    assert output.atomized is not None
    torch.testing.assert_close(output.coordinates[0, 2, 1], bb_ca[0, 2])
    assert not torch.any(output.coordinates[0, 2, 2:])
    assert output.residue_types[0, 2] == -1
    torch.testing.assert_close(output.complex_coordinates[0, 2, 1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(output.complex_coordinates[0, 3, 1], bb_ca[0, 3])
    assert not torch.any(output.complex_coordinates[0, 2:4, 2:])
    assert output.atomized.bond_mask[0, 2, 3]


class _FixedStateSampler(torch.nn.Module):
    """Returns a prepared final state instead of rolling one out."""

    group_keys = frozenset()

    def __init__(self, state, trunk_sequence_logits=None) -> None:
        super().__init__()
        self.state = state
        self.trunk_sequence_logits = trunk_sequence_logits

    def sample(self, **kwargs):
        return RFProteinaSamplingOutput(
            state=self.state,
            clean_prediction=self.state,
            insertion_logits=None,
            trajectory=[],
            trunk_sequence_logits=self.trunk_sequence_logits,
            seed=kwargs["seed"],
        )


class _FixedHead(torch.nn.Module):
    """An output head that returns chosen values, whatever the features are."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        return self.value.to(single.dtype)


class _RequiredAtomRecorder(torch.nn.Module):
    """A decoder that records the required atoms it is handed."""

    num_residue_types = 33
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.required = []

    def forward(self, backbone_ca, local_latents, mask, required_atom_mask=None, **kwargs):
        del backbone_ca, local_latents, kwargs
        self.required.append(required_atom_mask)
        return {
            "coords": torch.zeros(*mask.shape, 37, 3),
            "sequence_logits": torch.zeros(*mask.shape, 33),
            "residue_type": torch.ones_like(mask, dtype=torch.long) * mask,
            "atom_mask": mask.unsqueeze(-1).expand(-1, -1, 37).clone(),
        }


def _model_with_decoder(decoder: torch.nn.Module) -> RFProteina:
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    model.decoder = decoder
    return model


def test_rfproteina_decode_hands_the_required_atoms_to_the_decoder() -> None:
    decoder = _RequiredAtomRecorder()
    model = _model_with_decoder(decoder)
    state = {"bb_ca": torch.zeros(2, 3, 3), "local_latents": torch.zeros(2, 3, 4)}
    required = torch.rand(2, 3, 37) > 0.5

    model.decode(state, torch.ones(2, 3, dtype=torch.bool), required_atom_mask=required)

    assert len(decoder.required) == 1
    torch.testing.assert_close(decoder.required[0], required)

    # Tokens the autoencoder skips never reach the decoder, so their requirements do not either.
    decoder.required.clear()
    encoded = torch.tensor([[True, False, True], [False, True, False]])
    model.decode(state, torch.ones(2, 3, dtype=torch.bool), can_be_encoded=encoded, required_atom_mask=required)

    assert len(decoder.required) == 2
    torch.testing.assert_close(decoder.required[0], required[0:1, [0, 2]])
    torch.testing.assert_close(decoder.required[1], required[1:2, [1]])

    with pytest.raises(ValueError, match="required_atom_mask must have shape"):
        model.decode(state, torch.ones(2, 3, dtype=torch.bool), required_atom_mask=required[..., :36])


def test_rfproteina_prediction_requires_the_atoms_the_caller_conditions_on() -> None:
    decoder = _RequiredAtomRecorder()
    model = _model_with_decoder(decoder)
    mask = torch.ones(1, 3, dtype=torch.bool)
    static_features = _native_static_features(mask)
    state = {"bb_ca": torch.zeros(1, 3, 3), "local_latents": torch.zeros(1, 3, 4)}
    model.sampler = _FixedStateSampler(state)
    groups = static_features["condition_CoordinateGroup_1_atom"]
    groups[0, 0, [0, 1, 2]] = 0
    groups[0, 2, 4] = 3

    model(static_features, seed=3)

    torch.testing.assert_close(decoder.required[0], groups >= 0)
    assert decoder.required[0].sum() == 4

    decoder.required.clear()
    del static_features["condition_CoordinateGroup_1_atom"]
    model(static_features, seed=3)

    assert decoder.required == [None]


@pytest.mark.parametrize("sequence_head_policy", [False, True])
def test_rfproteina_prediction_applies_native_decoder_constraints(sequence_head_policy: bool) -> None:
    ala, gly, tyr = 1, 8, 19
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(
                input_schema="unified", num_residue_types=33, use_pair_bias=False, use_atom_presence=True
            ),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    ).cuda()
    model.config.sampling.condition_decoder_on_sequence_prediction = sequence_head_policy
    init_module_weights(model.decoder)
    # The decoder predicts TYR everywhere with no atoms but the reference one, and the trunk reads out GLY.
    decoder_logits = torch.full((1, 3, 33), -5.0, device="cuda")
    decoder_logits[..., tyr] = 5.0
    model.decoder.logit_linear = _FixedHead(decoder_logits)
    model.decoder.atom_presence_linear = _FixedHead(torch.full((1, 3, 37), -1.0, device="cuda"))
    trunk_logits = torch.full((1, 3, 33), -5.0, device="cuda")
    trunk_logits[..., gly] = 5.0
    state = {"bb_ca": torch.zeros(1, 3, 3, device="cuda"), "local_latents": torch.zeros(1, 3, 4, device="cuda")}
    model.sampler = _FixedStateSampler(state, trunk_logits)

    static_features = _native_static_features(torch.ones(1, 3, dtype=torch.bool, device="cuda"))
    static_features["condition_sequence_1_residue"] = torch.tensor([[ala, 0, 0]], device="cuda")
    static_features["mask_sequence_1_residue"] = torch.tensor([[True, False, False]], device="cuda")
    static_features["condition_CoordinateGroup_1_atom"][0, 0, [0, 1, 2, 3, 4]] = 0

    prediction = model(static_features, seed=5)

    # The conditioned token keeps ALA whatever the decoder says. The others take the decoder's TYR,
    # or the trunk's GLY when the sequence-head policy conditions the decoder on it.
    free = gly if sequence_head_policy else tyr
    assert prediction.residue_types.tolist() == [[ala, free, free]]
    # Every atom of ALA that the caller conditions on survives the head that denies it; the
    # others keep their reference atom only.
    expected = torch.zeros(1, 3, 37, dtype=torch.bool, device="cuda")
    expected[0, 0, [0, 1, 2, 3, 4]] = True
    expected[..., 1] = True
    torch.testing.assert_close(prediction.atom_mask, expected)


_ALA, _GLY, _TYR = 1, 8, 19
_BACKBONE = [0, 1, 2, 4]  # N, CA, C, O


def _unified_redesign_setup(num_steps: int = 3, chemistry: bool = False) -> tuple[RFProteina, dict[str, torch.Tensor]]:
    """A small unified-vocabulary model over four tokens, the second fixed to GLY and its backbone.

    The decoder predicts TYR everywhere and denies every atom but the reference one.
    """
    mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    residue_type = torch.tensor([[_ALA, _GLY, _TYR, _ALA]], device="cuda")
    atom_mask = get_residue_atom_mask(residue_type, 33, mask)
    coordinates = torch.randn(1, 4, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    static_features.update(
        {
            "residue_type": residue_type,
            "atom37_mask": atom_mask,
            "can_be_encoded": mask.clone(),
            "coords_nm": coordinates,
        }
    )
    fixed = torch.zeros_like(atom_mask)
    fixed[0, 1, _BACKBONE] = True
    static_features["mask_coordinate_1_atom"] = fixed
    static_features["condition_coordinate_1_atom"] = torch.where(fixed.unsqueeze(-1), coordinates * 10, 0.0)
    static_features["condition_CoordinateGroup_1_atom"][0, 1, _BACKBONE] = 0
    static_features["mask_sequence_1_residue"] = torch.tensor([[False, True, False, False]], device="cuda")
    static_features["condition_sequence_1_residue"] = residue_type.clone()

    model = (
        RFProteina(
            RFProteinaConfig(
                dtype="float32",
                trunk=_small_trunk_config(),
                encoder=_small_encoder_config(
                    input_schema="unified",
                    num_residue_types=33,
                    sequence_separation_bins=64,
                    use_pair_bias=False,
                    precompute_bias=False,
                    **(
                        {
                            "atom_chemistry": True,
                            "num_token_kinds": 7,
                            "num_elements": 9,
                            "element_embedding_dim": 4,
                            "num_atom_charges": 11,
                            "atom_charge_embedding_dim": 6,
                        }
                        if chemistry
                        else {}
                    ),
                ),
                decoder=_small_decoder_config(
                    input_schema="unified", num_residue_types=33, use_pair_bias=False, use_atom_presence=True
                ),
                sampling=_deterministic_sampling_config(num_steps=num_steps),
            )
        )
        .cuda()
        .eval()
    )
    init_module_weights(model)
    logits = torch.full((1, 4, 33), -5.0, device="cuda")
    logits[..., _TYR] = 5.0
    model.decoder.logit_linear = _FixedHead(logits)
    model.decoder.atom_presence_linear = _FixedHead(torch.full((1, 4, 37), -1.0, device="cuda"))
    return model, static_features


def test_rfproteina_encode_clean_state_reads_the_chemistry_of_each_atom() -> None:
    model, static_features = _unified_redesign_setup(chemistry=True)
    occupied = static_features["atom37_mask"]
    carbon = torch.where(occupied, 6, 0)
    nitrogen = carbon.clone()
    nitrogen[0, 1] = torch.where(occupied[0, 1], 7, 0)
    bonds = torch.zeros(1, 4, 37, 10, device="cuda")

    def encode(**chemistry):
        features = {**static_features, **chemistry}
        return model.encode_clean_state(features, sample_latent=False).mean

    with_carbon = encode(ground_truth_element=carbon, charge=torch.zeros_like(carbon), incident_bond_type_counts=bonds)
    with_nitrogen = encode(
        ground_truth_element=nitrogen, charge=torch.zeros_like(carbon), incident_bond_type_counts=bonds
    )
    # Upstream's batch spells the element ground_truth_element; BioIR's own name reaches the encoder just the same.
    renamed = encode(atom37_element=nitrogen, atom37_charge=torch.zeros_like(carbon), incident_bond_type_counts=bonds)

    assert not torch.allclose(with_carbon[0, 1], with_nitrogen[0, 1])
    torch.testing.assert_close(with_nitrogen, renamed, atol=0, rtol=0)
    # A batch without chemistry still encodes, on default chemistry.
    assert torch.isfinite(encode()).all()


def test_rfproteina_redesign_applies_native_decoder_constraints() -> None:
    model, static_features = _unified_redesign_setup()

    output = model.redesign(
        static_features, RFProteinaPartialSimulationConfig(partial_t_backbone=0.7, partial_t_latent=0.5), seed=7
    )

    assert output.partial_redesign is not None
    assert output.residue_types.tolist() == [[_TYR, _GLY, _TYR, _TYR]]
    expected = torch.zeros(1, 4, 37, dtype=torch.bool, device="cuda")
    expected[..., 1] = True
    expected[0, 1, _BACKBONE] = True
    torch.testing.assert_close(output.atom_mask, expected)
    torch.testing.assert_close(output.final_atom_mask, expected)


def test_rfproteina_redesign_by_fraction_integrates_the_tail_of_the_configured_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, static_features = _unified_redesign_setup(num_steps=8)
    # A densified tail is what separates the configured schedule's tail from one rebuilt around the start time.
    assert model.sampler.config is model.config.sampling
    for modality_config in (model.config.sampling.backbone, model.config.sampling.local_latents):
        modality_config.schedule.tail_steps = 2
        modality_config.schedule.tail_step_multiplier = 2
    rolled = []
    run = model.sampler._run

    def record(**kwargs):
        rolled.append(kwargs["schedules"])
        return run(**kwargs)

    monkeypatch.setattr(model.sampler, "_run", record)

    output = model.redesign(
        static_features, RFProteinaPartialSimulationConfig(partial_trajectory_fraction_steps=0.5), seed=7
    )

    assert output.partial_redesign is not None and output.partial_redesign.num_steps == 4
    for modality, modality_config in (
        ("bb_ca", model.config.sampling.backbone),
        ("local_latents", model.config.sampling.local_latents),
    ):
        full = create_modality_schedule(modality_config, 8, torch.device("cuda"))
        torch.testing.assert_close(rolled[0][modality].times, full.times[4:], rtol=0, atol=0)
        torch.testing.assert_close(rolled[0][modality].noise_rate, full.noise_rate[4:], rtol=0, atol=0)
        assert output.partial_redesign.start_times[modality] == full.times[4].item()


def test_rfproteina_sequence_head_fill_stays_within_the_token_kind_of_a_wide_decoder() -> None:
    vocabulary = residue_vocabulary(147)
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(
                input_schema="unified",
                num_residue_types=147,
                num_conditional_residue_types=147,
                num_token_kinds=7,
                atom37_occupancy=vocabulary.atom37_occupancy(),
                allowed_residue_classes=vocabulary.allowed_residue_types(7),
                use_pair_bias=False,
                use_atom_presence=True,
            ),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    # The trunk reads out a protein class for a protein, an RNA and a DNA token alike.
    trunk_logits = torch.full((1, 3, 147), -5.0)
    trunk_logits[..., 1] = 5.0
    padding_mask = torch.ones(1, 3, dtype=torch.bool)

    filled, conditioned = model._condition_on_predicted_sequence(
        trunk_logits, padding_mask, padding_mask, torch.tensor([[0, 1, 2]]), None, None
    )

    assert conditioned.all()
    assert filled[0, 0] == 1
    assert filled[0, 1] in vocabulary.allowed_residue_types(7)[1]
    assert filled[0, 2] in vocabulary.allowed_residue_types(7)[2]


def _nucleic_static_features() -> dict[str, torch.Tensor]:
    """Build protein plus fixed RNA/DNA context in unified tensor space."""
    padding_mask = torch.ones(1, 4, dtype=torch.bool)
    static_features = _native_static_features(padding_mask)
    static_features["token_type"] = torch.tensor([[0, 1, 2, 2]])
    static_features["residue_type"] = torch.tensor([[1, 22, 27, 30]])
    static_features["can_be_encoded"] = torch.ones_like(padding_mask)
    static_features["atom37_mask"] = get_residue_atom_mask(
        static_features["residue_type"],
        33,
        padding_mask,
    )
    nucleic_mask = static_features["token_type"] > 0
    coordinate_mask = static_features["atom37_mask"] & nucleic_mask.unsqueeze(-1)
    static_features["mask_coordinate_1_atom"] = coordinate_mask
    coordinate_values = torch.zeros(1, 4, 37, 3)
    coordinate_values[coordinate_mask] = torch.arange(
        int(coordinate_mask.sum()) * 3,
        dtype=torch.float32,
    ).reshape(-1, 3)
    static_features["condition_coordinate_1_atom"] = coordinate_values
    static_features["mask_sequence_1_residue"] = nucleic_mask
    static_features["condition_sequence_1_residue"] = static_features["residue_type"].clone()
    return static_features


def test_rfproteina_materializes_unified_nucleic_acid_output() -> None:
    static_features = _nucleic_static_features()
    # Leave the two DNA tokens generated while RNA remains fixed.
    static_features["mask_coordinate_1_atom"][:, 2:] = False
    static_features["mask_sequence_1_residue"][:, 2:] = False
    coordinates = torch.randn(1, 4, 37, 3)
    residue_types = torch.tensor([[2, 25, 29, 31]])
    atom_mask = get_residue_atom_mask(residue_types, 33, torch.ones(1, 4, dtype=torch.bool))

    output = materialize_nucleic_acid_output(
        coordinates,
        residue_types,
        atom_mask,
        static_features,
        decoder_num_residue_types=33,
    )

    assert output is not None
    torch.testing.assert_close(output.rna_token_mask, torch.tensor([[False, True, False, False]]))
    torch.testing.assert_close(output.dna_token_mask, torch.tensor([[False, False, True, True]]))
    assert output.residue_types[0, 1] == 22
    torch.testing.assert_close(output.residue_types[0, 2:], torch.tensor([29, 31]))
    selected = output.coordinate_condition_mask[0, 1]
    torch.testing.assert_close(
        output.coordinates[0, 1][selected],
        static_features["condition_coordinate_1_atom"][0, 1][selected] * 0.1,
    )
    torch.testing.assert_close(output.coordinates[0, 2], coordinates[0, 2] * output.atom_mask[0, 2].unsqueeze(-1))


def _unified_atom_mask(residue_types: torch.Tensor, width: int) -> torch.Tensor:
    """The decoder's own atom occupancy at ``width`` classes, as the sampler hands it to the output."""
    occupancy = None if width == 33 else residue_vocabulary(width).atom37_occupancy()
    return get_residue_atom_mask(residue_types, width, torch.ones_like(residue_types, dtype=torch.bool), occupancy)


@pytest.mark.parametrize("width", [33, 147])
def test_rfproteina_nucleic_acid_contract_accepts_every_unified_decoder(width: int) -> None:
    static_features = _nucleic_static_features()
    validate_nucleic_acid_contract(static_features, static_features["can_be_encoded"], decoder_num_residue_types=width)


def test_rfproteina_unified_nucleic_acid_output_does_not_depend_on_the_decoder_width() -> None:
    outputs = {}
    for width in (33, 147):
        static_features = _nucleic_static_features()
        # Leave the two DNA tokens generated while RNA remains fixed.
        static_features["mask_coordinate_1_atom"][:, 2:] = False
        static_features["mask_sequence_1_residue"][:, 2:] = False
        torch.manual_seed(0)
        coordinates = torch.randn(1, 4, 37, 3)
        residue_types = torch.tensor([[2, 25, 29, 31]])
        outputs[width] = materialize_nucleic_acid_output(
            coordinates,
            residue_types,
            _unified_atom_mask(residue_types, width),
            static_features,
            decoder_num_residue_types=width,
        )
    narrow, wide = outputs[33], outputs[147]
    assert narrow is not None and wide is not None
    for field in dataclasses.fields(narrow):
        a, b = getattr(narrow, field.name), getattr(wide, field.name)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(b, a, msg=lambda message, name=field.name: f"{name}: {message}")


@pytest.mark.parametrize("width", [21, 32])
def test_rfproteina_nucleic_acid_rejects_a_decoder_that_is_neither_legacy_nor_unified(width: int) -> None:
    static_features = _nucleic_static_features()
    with pytest.raises(ValueError, match="20-class decoder or a unified decoder"):
        validate_nucleic_acid_contract(
            static_features, static_features["can_be_encoded"], decoder_num_residue_types=width
        )
    residue_types = static_features["residue_type"]
    with pytest.raises(ValueError, match="must be 20 or at least 33"):
        materialize_nucleic_acid_output(
            torch.zeros(1, 4, 37, 3),
            residue_types,
            get_residue_atom_mask(residue_types, 33, torch.ones_like(residue_types, dtype=torch.bool)),
            static_features,
            decoder_num_residue_types=width,
        )


def test_wider_unified_vocabulary_keeps_nucleic_acids_where_the_33_class_decoder_has_them() -> None:
    """The invariant that lets one set of RNA/DNA ID ranges serve every unified decoder.

    If a wider vocabulary ever renumbered RNA or DNA, or gave them different atom occupancy, the
    output stage would validate and materialize the wrong residues without raising.
    """
    vocabulary = residue_vocabulary(147)
    rna = [index for index, group in enumerate(vocabulary.groups) if group == "rna"]
    dna = [index for index, group in enumerate(vocabulary.groups) if group == "dna"]
    assert rna == list(range(*RNA_RESIDUE_TYPE_RANGE))
    assert dna == list(range(*DNA_RESIDUE_TYPE_RANGE))
    wide = get_atom37_mask_table(None, 147, vocabulary.atom37_occupancy())
    narrow = get_atom37_mask_table(None, 33, None)
    torch.testing.assert_close(wide[rna + dna], narrow[rna + dna])


def test_rfproteina_legacy_contract_requires_nonencodable_nucleic_acids() -> None:
    static_features = _nucleic_static_features()
    with pytest.raises(ValueError, match="must be non-encodable"):
        validate_nucleic_acid_contract(
            static_features,
            static_features["can_be_encoded"],
            decoder_num_residue_types=20,
        )


def test_rfproteina_legacy_decoder_materializes_fixed_nucleic_context() -> None:
    static_features = _nucleic_static_features()
    state = {
        "bb_ca": torch.randn(1, 4, 3),
        "local_latents": torch.randn(1, 4, 4),
    }

    class StaticSampler(torch.nn.Module):
        group_keys = frozenset()

        def __init__(self) -> None:
            super().__init__()
            self.can_be_encoded = None

        def sample(self, **kwargs):
            self.can_be_encoded = kwargs["can_be_encoded"]
            return RFProteinaSamplingOutput(
                state=state,
                clean_prediction=state,
                insertion_logits=None,
                trajectory=[],
                seed=kwargs["seed"],
            )

    class LegacyDecoder(torch.nn.Module):
        num_residue_types = 20
        dtype = torch.float32

        def forward(self, backbone_ca, local_latents, mask, **kwargs):
            del backbone_ca, local_latents, kwargs
            return {
                "coords": torch.full((*mask.shape, 37, 3), 7.0),
                "sequence_logits": torch.zeros(*mask.shape, 20),
                "residue_type": torch.zeros_like(mask, dtype=torch.long),
            }

    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    sampler = StaticSampler()
    model.sampler = sampler
    model.decoder = LegacyDecoder()

    prediction = model(static_features, seed=17)

    assert prediction.nucleic_acid is not None
    torch.testing.assert_close(
        sampler.can_be_encoded,
        torch.tensor([[True, False, False, False]]),
    )
    # Raw decoder outputs stay available for parity/debugging.
    assert prediction.residue_types[0, 0] == 0
    assert torch.all(prediction.residue_types[0, 1:] == -1)
    # The canonical output combines 1-based unified protein IDs with the fixed
    # RNA/DNA identities from the tensorized input.
    torch.testing.assert_close(prediction.complex_residue_types, torch.tensor([[1, 22, 27, 30]]))
    fixed = static_features["mask_coordinate_1_atom"].bool()
    torch.testing.assert_close(
        prediction.complex_coordinates[fixed],
        static_features["condition_coordinate_1_atom"][fixed] * 0.1,
    )
    torch.testing.assert_close(
        prediction.complex_atom_mask[0, 1:],
        static_features["atom37_mask"][0, 1:],
    )


def test_rfproteina_rejects_generated_nucleic_acids_with_legacy_decoder() -> None:
    static_features = _nucleic_static_features()
    static_features["mask_coordinate_1_atom"][0, 2] = False

    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    with pytest.raises(ValueError, match="legacy 20-class protein decoder"):
        model(static_features, seed=3)


def test_rfproteina_latest_schema_consumes_na_secondary_structure_conditions() -> None:
    static_features = _nucleic_static_features()
    static_features["na_ss_state"] = torch.tensor([[0, 1, 2, 0]])
    static_features["na_ss_state_mask"] = torch.tensor([[False, True, False, False]])
    static_features["na_ss_pair"] = torch.zeros(1, 4, 4)
    static_features["na_ss_pair"][0, 1, 2] = 1
    static_features["na_ss_pair_mask"] = torch.zeros(1, 4, 4, dtype=torch.bool)
    static_features["na_ss_pair_mask"][0, 1, 2] = True
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(num_residue_types=33),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    with pytest.raises(ValueError, match="not consumed"):
        validate_nucleic_acid_contract(
            static_features,
            static_features["can_be_encoded"],
            decoder_num_residue_types=33,
        )
    validate_nucleic_acid_contract(
        static_features,
        static_features["can_be_encoded"],
        decoder_num_residue_types=33,
        supports_na_secondary_structure=True,
    )
    model = model.cuda()
    static_features = {name: value.cuda() for name, value in static_features.items()}

    state = {
        "bb_ca": torch.zeros(1, 4, 3, device="cuda"),
        "local_latents": torch.zeros(1, 4, 4, device="cuda"),
    }
    times = {"bb_ca": torch.zeros(1, device="cuda"), "local_latents": torch.zeros(1, device="cuda")}
    self_conditioning = {name: torch.zeros_like(value) for name, value in state.items()}
    projection_inputs = {}

    def capture(name):
        def hook(_module, args):
            projection_inputs[name] = args[0]

        return hook

    handles = [
        model.representation_builder.single_projection.register_forward_pre_hook(capture("single")),
        model.representation_builder.pair_projection.register_forward_pre_hook(capture("pair")),
    ]
    model.representation_builder(
        static_features,
        state,
        times,
        self_conditioning,
        False,
        static_features["padding_mask"],
    )
    for handle in handles:
        handle.remove()

    config = model.config.representation
    na_state_offset = (
        3
        + config.latent_dim
        + config.token_type_embedding_dim
        + config.num_conditional_residue_types
        + 2 * 37 * 4
        + 37
        + config.secondary_structure_embedding_dim
        + 2 * 37
        + 37
        + 2 * 37
    )
    expected_state = torch.stack(
        (static_features["na_ss_state"].float(), static_features["na_ss_state_mask"].float()),
        dim=-1,
    )
    torch.testing.assert_close(projection_inputs["single"][..., na_state_offset : na_state_offset + 2], expected_state)
    torch.testing.assert_close(
        projection_inputs["pair"][..., -2:],
        torch.stack(
            (static_features["na_ss_pair"], static_features["na_ss_pair_mask"].float()),
            dim=-1,
        ),
    )


def test_rfproteina_hungarian_assignment_is_distinct_and_chain_local() -> None:
    logits = torch.zeros(1, 9, 9)
    logits[0, 4, [0, 1]] = torch.tensor([10.0, 9.0])
    logits[0, 5, [0, 1]] = torch.tensor([10.0, 0.0])
    logits[0, 6, [2, 3]] = torch.tensor([1.0, 4.0])
    logits.requires_grad_()
    padding_mask = torch.tensor([[True] * 8 + [False]])
    index_mask = torch.ones(1, 9, 37, dtype=torch.bool)
    atom37_mask = torch.zeros_like(index_mask)
    atom37_mask[:, :8, 1] = True
    index_mask[:, 4:8, 1] = False
    chain_iid = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 2, -1]])

    assignment = assign_unindexed_tokens(logits, padding_mask, index_mask, chain_iid, atom37_mask)

    torch.testing.assert_close(
        assignment,
        torch.tensor(
            [
                [
                    UNASSIGNED_TOKEN_INDEX,
                    UNASSIGNED_TOKEN_INDEX,
                    UNASSIGNED_TOKEN_INDEX,
                    UNASSIGNED_TOKEN_INDEX,
                    1,
                    0,
                    3,
                    UNASSIGNED_TOKEN_INDEX,
                    UNASSIGNED_TOKEN_INDEX,
                ]
            ]
        ),
    )
    assert not assignment.requires_grad


def test_rfproteina_materializes_assigned_motif_conditions() -> None:
    padding_mask = torch.ones(1, 6, dtype=torch.bool)
    static_features = _native_static_features(padding_mask)
    index_mask = torch.ones(1, 6, 37, dtype=torch.bool)
    index_mask[:, 4:] = False
    static_features["mask_index_1_residue"] = index_mask

    coordinate_mask = torch.zeros(1, 6, 37, dtype=torch.bool)
    coordinate_mask[0, 0, 1] = True
    coordinate_mask[0, 4, 36] = True
    static_features["mask_coordinate_1_atom"] = coordinate_mask
    static_features["condition_coordinate_1_atom"] = torch.zeros(1, 6, 37, 3)
    static_features["condition_coordinate_1_atom"][0, 0, 1] = torch.tensor([20.0, 30.0, 40.0])
    static_features["condition_coordinate_1_atom"][0, 4, 36] = torch.tensor([10.0, 20.0, 30.0])

    sequence_mask = torch.zeros(1, 6, dtype=torch.bool)
    sequence_mask[0, [0, 4, 5]] = True
    static_features["mask_sequence_1_residue"] = sequence_mask
    static_features["condition_sequence_1_residue"] = torch.tensor([[9, 0, 0, 0, 7, 8]])

    coordinates = torch.randn(1, 6, 37, 3)
    raw_coordinates = coordinates.clone()
    residue_types = torch.tensor([[1, 2, 3, 4, 5, 6]])
    assignment = torch.tensor([[-1, -1, -1, -1, 1, 2]])

    insertion = apply_insertion_assignment(coordinates, residue_types, assignment, static_features)

    torch.testing.assert_close(
        insertion.generated_token_mask,
        torch.tensor([[True, True, True, True, False, False]]),
    )
    torch.testing.assert_close(
        insertion.source_token_index,
        torch.tensor([[-1, 4, 5, -1, -1, -1]]),
    )
    torch.testing.assert_close(
        insertion.target_mask,
        torch.tensor([[False, True, True, False, False, False]]),
    )
    torch.testing.assert_close(insertion.coordinates[0, 0, 1], torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(insertion.coordinates[0, 1, 36], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(insertion.coordinates[0, 1, 0], coordinates[0, 1, 0])
    torch.testing.assert_close(insertion.residue_types, torch.tensor([[9, 7, 8, 4, 5, 6]]))
    assert insertion.coordinate_mask[0, 0, 1]
    assert insertion.coordinate_mask[0, 1, 36]
    torch.testing.assert_close(
        insertion.sequence_mask,
        torch.tensor([[True, True, True, False, False, False]]),
    )
    torch.testing.assert_close(coordinates, raw_coordinates)
    torch.testing.assert_close(residue_types, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    legacy_insertion = apply_insertion_assignment(
        coordinates,
        residue_types,
        assignment,
        static_features,
        condition_sequence_offset=-1,
    )
    torch.testing.assert_close(legacy_insertion.residue_types, torch.tensor([[8, 6, 7, 4, 5, 6]]))


def test_rfproteina_aligns_assigned_internal_coordinate_group() -> None:
    padding_mask = torch.ones(1, 6, dtype=torch.bool)
    static_features = _native_static_features(padding_mask)
    atom37_mask = torch.zeros(1, 6, 37, dtype=torch.bool)
    atom37_mask[:, :, 1] = True
    static_features["atom37_mask"] = atom37_mask
    index_mask = torch.ones(1, 6, 37, dtype=torch.bool)
    index_mask[:, 3:, 1] = False
    static_features["mask_index_1_residue"] = index_mask

    internal_mask = torch.zeros_like(atom37_mask)
    internal_mask[:, 3:, 1] = True
    static_features["mask_InternalCoordinate_1_atom"] = internal_mask
    static_features["condition_InternalCoordinate_1_atom"] = torch.zeros(1, 6, 37, 3)
    static_features["condition_InternalCoordinate_1_atom"][0, 3:, 1] = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    )
    coordinate_groups = torch.full((1, 6, 37), -1, dtype=torch.long)
    coordinate_groups[:, 3:, 1] = 1
    static_features["condition_CoordinateGroup_1_atom"] = coordinate_groups

    coordinates = torch.zeros(1, 6, 37, 3)
    generated_group = torch.tensor([[2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [1.0, 3.0, 0.0]])
    coordinates[0, :3, 1] = generated_group
    residue_types = torch.zeros(1, 6, dtype=torch.long)
    assignment = torch.tensor([[-1, -1, -1, 0, 1, 2]])

    insertion = apply_insertion_assignment(coordinates, residue_types, assignment, static_features)

    torch.testing.assert_close(insertion.coordinates[0, :3, 1], generated_group, atol=1e-6, rtol=1e-6)
    assert torch.all(insertion.coordinate_mask[0, :3, 1])
    assert not torch.any(insertion.coordinate_mask[0, 3:, 1])


def test_rfproteina_velocity_trunk_cudagraph_matches_eager() -> None:
    """Graphing the velocity trunk must not change the rollout.

    The trunk sees identical shapes on every sampling step, so one capture
    serves the whole rollout. The run must be long enough to get past the
    tracker's warmup thresholds, otherwise this would silently only exercise
    the eager fallback -- hence the explicit capture assertion.
    """
    trunk = _small_trunk_config()
    sampling = _deterministic_sampling_config(num_steps=8)
    mask = torch.tensor([[True] * 8 + [False] * 2], device="cuda")
    torch.manual_seed(3)
    single = torch.randn(1, 10, trunk.token_dim, device="cuda")
    pair = torch.randn(1, 10, 10, trunk.pair_dim, device="cuda")
    condition = torch.randn(1, 10, trunk.condition_dim, device="cuda")

    def build_representations(state, times, self_conditioning, active):
        del state, times, self_conditioning, active
        return RFProteinaRepresentations(single=single, pair=pair, condition=condition)

    def build_model() -> RFProteina:
        torch.manual_seed(11)
        model = (
            RFProteina(
                RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config(), sampling=sampling)
            )
            .cuda()
            .eval()
        )
        init_module_weights(model)
        return model

    eager = build_model()
    graphed = build_model()
    graphed.optimize(
        {
            "velocity_model": AcceleratedConfig(
                backend=BackendType.TORCH,
                default=BaseConfig(
                    graph_optimization_config=CUDAGraphOptimizationConfig(
                        graph_optimization_mode=(GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH),
                        # GRAPH_VERIFIED is only reachable when the runtime is
                        # asked to verify; the default leaves capture at
                        # GRAPH_CAPTURED. Assert the stronger state and enable
                        # the check that produces it.
                        verify_capture=True,
                    )
                ),
            )
        }
    )

    reference = eager.sample(mask, build_representations, seed=5)
    captured = graphed.sample(mask, build_representations, seed=5)

    tracker = graphed.velocity_model.trunk_graph.tracker
    assert isinstance(tracker, CUDAGraphOptimizationTracker)
    states = list(tracker.graph_state_by_key.values())
    assert len(states) == 1, "one shape across the rollout means one graph, whatever its depths"
    assert states[0].preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert not tracker.fallback_to_eager_by_key, "captured, not run eagerly"

    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(captured.state[modality], reference.state[modality], atol=1e-5, rtol=1e-4)


def test_rfproteina_kf_trunk_cudagraph_replays_are_stable() -> None:
    """A graphed KF trunk must replay the same answer every time, and match eager.

    The sibling test above runs a float32 trunk too narrow for the fused TriMul
    KF chain, so it exercises the regular path only. This one picks widths the
    KF configs ship for and turns on token padding, which is what selects the
    chain.

    This covers KF-path graph parity, not launch-ordering races: those depend
    on timing, so a small trunk can pass while a wide one fails. The invariant
    that keeps them away lives in
    ``tests/ops/test_trimul_kf.py::test_source_kernels_never_request_programmatic_dependent_launch``.
    """
    trunk = _small_trunk_config().model_copy(
        update={"dtype": "bfloat16", "pair_dim": 32, "trimul_hidden_dim": 32, "enable_token_pad": True}
    )
    sampling = _deterministic_sampling_config(num_steps=6)
    mask = torch.ones(1, 16, dtype=torch.bool, device="cuda")
    torch.manual_seed(5)
    single = torch.randn(1, 16, trunk.token_dim, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(1, 16, 16, trunk.pair_dim, device="cuda", dtype=torch.bfloat16)
    condition = torch.randn(1, 16, trunk.condition_dim, device="cuda", dtype=torch.bfloat16)

    def build_representations(state, times, self_conditioning, active):
        del state, times, self_conditioning, active
        return RFProteinaRepresentations(single=single, pair=pair, condition=condition)

    def build_model() -> RFProteina:
        torch.manual_seed(11)
        model = (
            RFProteina(
                RFProteinaConfig(dtype="bfloat16", trunk=trunk, decoder=_small_decoder_config(), sampling=sampling)
            )
            .cuda()
            .eval()
        )
        init_module_weights(model)
        return model

    eager = build_model()
    if not any(isinstance(module, TriangleMultiplicationNode) and module.kf_available for module in eager.modules()):
        pytest.skip("the fused TriMul KF chain does not ship for this GPU or these widths")

    graphed = build_model()
    graphed.configure_graphs(capacity=4, verify_capture=True)

    reference = eager.sample(mask, build_representations, seed=5)
    first = graphed.sample(mask, build_representations, seed=5)
    second = graphed.sample(mask, build_representations, seed=5)

    status = graphed.graph_status()
    assert status["captured"] >= 1, "the trunk must replay from a graph, not run eagerly"
    assert status["fallback"] == 0, f"capture fell back to eager: {status}"

    for modality in ("bb_ca", "local_latents"):
        assert torch.equal(first.state[modality], second.state[modality]), (
            f"two replays of the same input disagree on {modality}"
        )
        torch.testing.assert_close(first.state[modality], reference.state[modality], atol=1e-5, rtol=1e-4)


def test_rfproteina_model_level_sampling_smoke() -> None:
    trunk = _small_trunk_config()
    sampling = _deterministic_sampling_config(num_steps=2)
    model = RFProteina(
        RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config(), sampling=sampling)
    ).cuda()
    init_module_weights(model)
    mask = torch.tensor([[True] * 8 + [False] * 2], device="cuda")
    single = torch.randn(1, 10, trunk.token_dim, device="cuda")
    pair = torch.randn(1, 10, 10, trunk.pair_dim, device="cuda")
    condition = torch.randn(1, 10, trunk.condition_dim, device="cuda")
    active_flags: list[bool] = []

    def build_representations(state, times, self_conditioning, active):
        del state, times, self_conditioning
        active_flags.append(active)
        return RFProteinaRepresentations(single=single, pair=pair, condition=condition)

    output = model({"padding_mask": mask}, build_representations)
    assert isinstance(output, RFProteinaPredictionOutput)
    assert output.sampling.state["bb_ca"].shape == (1, 10, 3)
    assert output.sampling.state["local_latents"].shape == (1, 10, 4)
    assert torch.isfinite(output.sampling.state["bb_ca"]).all()
    assert torch.isfinite(output.sampling.state["local_latents"]).all()
    assert output.coordinates.shape == (1, 10, 37, 3)
    assert output.sequence_logits.shape == (1, 10, 20)
    assert output.residue_types.shape == (1, 10)
    assert output.atom_mask.shape == (1, 10, 37)
    assert output.complex_residue_types.shape == (1, 10)
    assert output.complex_atom_mask.shape == (1, 10, 37)
    assert output.insertion_assignment is None
    decoded = model.decode(output.sampling.state, mask)
    torch.testing.assert_close(output.coordinates, decoded["coords"])
    torch.testing.assert_close(output.sequence_logits, decoded["sequence_logits"])
    torch.testing.assert_close(output.residue_types, decoded["residue_type"])
    assert active_flags == [False, True]


def test_rfproteina_model_existing_protein_redesign_is_reproducible_and_restores_fixed_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    legacy_residue_type = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda")
    residue_type = legacy_residue_type + 1
    atom_mask = get_residue_atom_mask(legacy_residue_type, 20, mask)
    coordinates = torch.randn(1, 6, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    static_features.update(
        {
            "residue_type": residue_type,
            "atom37_mask": atom_mask,
            "can_be_encoded": mask.clone(),
            "coords_nm": coordinates,
        }
    )
    fixed_mask = torch.zeros_like(atom_mask)
    fixed_mask[0, 2, 1] = True
    static_features["mask_coordinate_1_atom"] = fixed_mask
    conditioned_angstrom = torch.zeros_like(coordinates)
    conditioned_angstrom[0, 2, 1] = torch.tensor([14.0, 25.0, 36.0], device="cuda")
    static_features["condition_coordinate_1_atom"] = conditioned_angstrom
    static_features["mask_index_1_residue"] = atom_mask.clone()
    sequence_mask = torch.zeros_like(mask)
    sequence_mask[0, 2] = True
    static_features["mask_sequence_1_residue"] = sequence_mask
    conditioned_sequence = residue_type.clone()
    conditioned_sequence[0, 2] = 20
    static_features["condition_sequence_1_residue"] = conditioned_sequence

    model = (
        RFProteina(
            RFProteinaConfig(
                dtype="float32",
                trunk=_small_trunk_config(),
                encoder=_small_encoder_config(),
                decoder=_small_decoder_config(),
                sampling=_deterministic_sampling_config(num_steps=3),
            )
        )
        .cuda()
        .eval()
    )
    init_module_weights(model)
    partial_config = RFProteinaPartialSimulationConfig(
        partial_t_backbone=0.7,
        partial_t_latent=0.5,
    )

    first = model.redesign(static_features, partial_config, seed=101)
    second = model.redesign(static_features, partial_config, seed=101)
    other = model.redesign(static_features, partial_config, seed=102)

    assert first.partial_redesign is not None
    assert first.insertion is not None
    assert first.insertion.residue_types[0, 2] == 19
    # The raw prediction can match the conditioned class by chance.
    torch.testing.assert_close(first.residue_types, first.sequence_logits.argmax(dim=-1))
    assert first.partial_redesign.num_steps == 3
    assert first.partial_redesign.start_times == {
        "bb_ca": 0.7,
        "local_latents": 0.5,
    }
    assert first.sampling.seed == 101
    torch.testing.assert_close(
        first.partial_redesign.clean_state["bb_ca"],
        coordinates[..., 1, :],
    )
    torch.testing.assert_close(first.final_coordinates[fixed_mask], conditioned_angstrom[fixed_mask] * 0.1)
    torch.testing.assert_close(first.final_coordinates, first.partial_redesign.coordinates)
    expected_residue_types = torch.where(first.residue_types >= 0, first.residue_types + 1, first.residue_types)
    torch.testing.assert_close(first.final_residue_types, expected_residue_types)
    torch.testing.assert_close(first.final_atom_mask, first.atom_mask)
    torch.testing.assert_close(first.final_coordinates, second.final_coordinates)
    assert not torch.allclose(first.sampling.state["bb_ca"], other.sampling.state["bb_ca"])

    supplied_clean = first.partial_redesign.clean_state

    def reject_encoder(*_args, **_kwargs):
        raise AssertionError("supplied clean state must bypass the encoder")

    monkeypatch.setattr(model, "encode_clean_state", reject_encoder)
    supplied = model.redesign(static_features, partial_config, clean_state=supplied_clean, seed=103)
    assert supplied.partial_redesign is not None
    with pytest.raises(ValueError, match="either coordinates or clean_state"):
        model.redesign(static_features, partial_config, coordinates=coordinates, clean_state=supplied_clean)


def test_rfproteina_inverse_fold_restores_backbone_and_varies_latents() -> None:
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    static_features = _native_static_features(mask)
    legacy_residue_type = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda")
    residue_type = legacy_residue_type + 1
    atom_mask = get_residue_atom_mask(legacy_residue_type, 20, mask)
    coordinates = torch.randn(1, 6, 37, 3, device="cuda") * atom_mask.unsqueeze(-1)
    static_features.update(
        {
            "residue_type": residue_type,
            "atom37_mask": atom_mask,
            "can_be_encoded": mask.clone(),
            "coords_nm": coordinates,
        }
    )
    model = (
        RFProteina(
            RFProteinaConfig(
                dtype="float32",
                trunk=_small_trunk_config(),
                encoder=_small_encoder_config(),
                decoder=_small_decoder_config(),
                sampling=_deterministic_sampling_config(num_steps=3),
            )
        )
        .cuda()
        .eval()
    )
    init_module_weights(model)

    first = model.inverse_fold(static_features, coordinates=coordinates, seed=11)
    second = model.inverse_fold(static_features, coordinates=coordinates, seed=11)
    other = model.inverse_fold(static_features, coordinates=coordinates, seed=12)

    assert first.partial_redesign is not None
    assert first.partial_redesign.start_times == {"bb_ca": 0.0, "local_latents": 0.0}
    assert first.partial_redesign.num_steps == 3
    torch.testing.assert_close(first.sampling.state["bb_ca"], first.partial_redesign.clean_state["bb_ca"])
    torch.testing.assert_close(first.sampling.state["bb_ca"], second.sampling.state["bb_ca"])
    torch.testing.assert_close(first.sampling.state["local_latents"], second.sampling.state["local_latents"])
    assert not torch.allclose(first.sampling.state["local_latents"], other.sampling.state["local_latents"])


@pytest.mark.parametrize(
    ("motif_positions", "is_unindexed_context"),
    [((1, 2), False), ((6, 7), True)],
    ids=["indexed", "unindexed-context"],
)
def test_rfproteina_model_native_motif_prediction_smoke(
    motif_positions: tuple[int, int],
    is_unindexed_context: bool,
) -> None:
    trunk = _small_trunk_config()
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=trunk,
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=2),
        )
    ).cuda()
    init_module_weights(model)
    mask = torch.tensor([[True] * 8 + [False] * 2], device="cuda")
    static_features = _native_static_features(mask)
    coordinate_mask = torch.zeros(1, 10, 37, dtype=torch.bool, device="cuda")
    coordinate_mask[:, motif_positions, 36] = True
    static_features["mask_coordinate_1_atom"] = coordinate_mask
    static_features["condition_coordinate_1_atom"] = torch.randn(1, 10, 37, 3, device="cuda")
    sequence_mask = torch.zeros_like(mask)
    sequence_mask[:, motif_positions] = True
    static_features["mask_sequence_1_residue"] = sequence_mask
    static_features["condition_sequence_1_residue"] = torch.full((1, 10), 5, dtype=torch.long, device="cuda")
    index_mask = torch.ones(1, 10, 37, dtype=torch.bool, device="cuda")
    if is_unindexed_context:
        index_mask[:, motif_positions] = False
    static_features["mask_index_1_residue"] = index_mask

    output = model(static_features, seed=17)

    assert output.sampling.seed == 17
    assert output.coordinates.shape == (1, 10, 37, 3)
    assert output.sequence_logits.shape == (1, 10, 20)
    assert output.insertion_logits is output.sampling.insertion_logits
    assert output.insertion_logits is not None
    assert output.insertion_logits.shape == (1, 10, 10)
    assert torch.isfinite(output.coordinates).all()
    assert torch.isfinite(output.sequence_logits).all()
    assert torch.isfinite(output.insertion_logits).all()
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    assert torch.count_nonzero(output.insertion_logits[~pair_mask]) == 0
    assert output.insertion_assignment is not None
    assert output.insertion is not None
    if is_unindexed_context:
        assigned = output.insertion_assignment[0, list(motif_positions)]
        assert torch.all((assigned >= 0) & (assigned < 6))
        assert torch.unique(assigned).numel() == len(motif_positions)
        query_mask = torch.zeros_like(output.insertion_assignment, dtype=torch.bool)
        query_mask[:, motif_positions] = True
        assert torch.all(output.insertion_assignment[~query_mask] == -1)
        assert not torch.any(output.insertion.generated_token_mask[:, motif_positions])
    else:
        assert torch.all(output.insertion_assignment == -1)
        assigned = torch.tensor(motif_positions, device="cuda")

    for source, target in zip(motif_positions, assigned.tolist(), strict=True):
        torch.testing.assert_close(
            output.coordinates_with_insertion[0, target, 36],
            static_features["condition_coordinate_1_atom"][0, source, 36] / 10.0,
        )
        # Static features use unified residue IDs (mask at 0), while this
        # small decoder has the legacy 20-class output space.
        assert output.residue_types_with_insertion[0, target] == 4
        assert output.insertion.coordinate_mask[0, target, 36]
        assert output.insertion.sequence_mask[0, target]


def test_rfproteina_resolves_motif_runtime_masks_from_static_features() -> None:
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    mask = torch.tensor([[True, True, True, True, False]])
    static_features = _native_static_features(mask)
    static_features["can_be_encoded"] = torch.tensor([[True, False, True, True, True]])
    static_features["has_known_index"] = torch.tensor([[True, True, False, True, True]])
    coordinate_mask = torch.zeros(1, 5, 37, dtype=torch.bool)
    coordinate_mask[0, 1, 36] = True
    coordinate_mask[0, 4, 0] = True
    static_features["mask_coordinate_1_atom"] = coordinate_mask

    class RecordingSampler:
        group_keys = frozenset()

        def __init__(self) -> None:
            self.can_be_encoded = None
            self.coordinate_condition_mask = None

        def sample(self, predict, padding_mask, dtype, **kwargs):
            del predict, dtype
            self.can_be_encoded = kwargs["can_be_encoded"]
            self.coordinate_condition_mask = kwargs["coordinate_condition_mask"]
            state = {
                "bb_ca": torch.zeros(*padding_mask.shape, 3),
                "local_latents": torch.zeros(*padding_mask.shape, 4),
            }
            return RFProteinaSamplingOutput(
                state=state,
                clean_prediction=state,
                insertion_logits=None,
                trajectory=[],
                seed=kwargs["seed"],
            )

    recorder = RecordingSampler()
    model.sampler = recorder
    model.sample(mask, static_features=static_features)

    torch.testing.assert_close(
        recorder.can_be_encoded,
        torch.tensor([[True, False, False, True, False]]),
    )
    torch.testing.assert_close(
        recorder.coordinate_condition_mask,
        torch.tensor([[False, True, False, False, False]]),
    )

    explicit_can_be_encoded = torch.tensor([[False, True, False, False, True]])
    explicit_coordinate_mask = torch.tensor([[False, False, True, False, True]])
    model.sample(
        mask,
        static_features=static_features,
        can_be_encoded=explicit_can_be_encoded,
        coordinate_condition_mask=explicit_coordinate_mask,
    )

    torch.testing.assert_close(
        recorder.can_be_encoded,
        torch.tensor([[False, True, False, False, False]]),
    )
    torch.testing.assert_close(
        recorder.coordinate_condition_mask,
        torch.tensor([[False, False, True, False, False]]),
    )

    static_features["mask_coordinate_1_atom"] = torch.zeros(1, 5, 36, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask_coordinate_1_atom"):
        model.sample(mask, static_features=static_features)


class _MaskRecordingSampler:
    """Stands in for the sampler and keeps the masks the model hands it."""

    group_keys = frozenset()

    def __init__(self) -> None:
        self.padding_mask = None
        self.can_be_encoded = None

    def sample(self, predict, padding_mask, dtype, **kwargs):
        del predict, dtype
        self.padding_mask = padding_mask
        self.can_be_encoded = kwargs["can_be_encoded"]
        state = {"bb_ca": torch.zeros(*padding_mask.shape, 3), "local_latents": torch.zeros(*padding_mask.shape, 4)}
        return RFProteinaSamplingOutput(
            state=state, clean_prediction=state, insertion_logits=None, trajectory=[], seed=kwargs["seed"]
        )


def test_rfproteina_flow_backbone_mask_is_the_rollout_mask_and_narrows_latents() -> None:
    """A conditioning-only token stays a real token for the trunk but leaves the rollout.

    The sampler takes its generated-token mask where it takes the padding mask, and latents are emitted
    only where the backbone is, so a caller that supplies one mask cannot leave the other disagreeing.
    """
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    mask = torch.tensor([[True, True, True, True, False]])
    static_features = _native_static_features(mask)
    static_features["flow_backbone_mask"] = torch.tensor([[True, False, True, True, True]])
    recorder = _MaskRecordingSampler()
    model.sampler = recorder

    model.sample(mask, static_features=static_features)

    # Token 1 only conditions; token 4 is padding, which the backbone mask cannot bring back.
    torch.testing.assert_close(recorder.padding_mask, torch.tensor([[True, False, True, True, False]]))
    torch.testing.assert_close(recorder.can_be_encoded, torch.tensor([[True, False, True, True, False]]))

    del static_features["flow_backbone_mask"]
    model.sample(mask, static_features=static_features)
    torch.testing.assert_close(recorder.padding_mask, mask)
    torch.testing.assert_close(recorder.can_be_encoded, mask)

    static_features["flow_backbone_mask"] = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="flow_backbone_mask"):
        model.sample(mask, static_features=static_features)


def test_rfproteina_reference_state_centers_over_the_generated_tokens_only() -> None:
    from bionemo_ir._torch.modules.rfproteina.sampler import sample_reference_state

    config = RFProteinaSamplingConfig(num_steps=3)
    config.backbone.reference.zero_com_noise = True
    config.backbone.reference.stochastic_centering_mode = "none"  # the offset would move the centre on purpose
    generated = torch.tensor([[True, False, True, True, True, True, False]], device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7)

    state = sample_reference_state(generated, 4, torch.float32, config, generator=generator)

    assert (state["bb_ca"][:, ~generated[0]] == 0).all(), "a conditioning-only token starts at the origin"
    torch.testing.assert_close(
        state["bb_ca"][0, generated[0]].mean(0), torch.zeros(3, device="cuda"), atol=1e-6, rtol=0
    )


def test_rfproteina_conditioning_only_tokens_hold_zero_state_through_a_rollout() -> None:
    trunk = _small_trunk_config()
    sampling = _deterministic_sampling_config(num_steps=3)
    sampling.backbone.reference.zero_com_noise = True
    model = RFProteina(
        RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config(), sampling=sampling)
    ).cuda()
    init_module_weights(model)
    tokens, conditioning_only = 6, 2
    mask = torch.ones(1, tokens, dtype=torch.bool, device="cuda")
    generated = mask.clone()
    generated[0, conditioning_only] = False
    single = torch.randn(1, tokens, trunk.token_dim, device="cuda")
    pair = torch.randn(1, tokens, tokens, trunk.pair_dim, device="cuda")
    condition = torch.randn(1, tokens, trunk.condition_dim, device="cuda")
    seen: list[dict[str, torch.Tensor]] = []

    def build_representations(state, times, self_conditioning, active):
        del times, self_conditioning, active
        seen.append({name: value.clone() for name, value in state.items()})
        return RFProteinaRepresentations(single=single, pair=pair, condition=condition)

    held = model({"padding_mask": mask, "flow_backbone_mask": generated}, build_representations)
    held_states, seen[:] = list(seen), []
    free = model({"padding_mask": mask}, build_representations)

    assert len(held_states) == sampling.num_steps
    for snapshot in held_states:
        # The trunk is handed every token on every step, and the held one never leaves the origin.
        assert snapshot["bb_ca"].shape == (1, tokens, 3)
        assert (snapshot["bb_ca"][0, conditioning_only] == 0).all()
        assert (snapshot["local_latents"][0, conditioning_only] == 0).all()
        assert (snapshot["bb_ca"][0, generated[0]] != 0).any()
    torch.testing.assert_close(
        held_states[0]["bb_ca"][0, generated[0]].mean(0), torch.zeros(3, device="cuda"), atol=1e-6, rtol=0
    )
    assert (held.sampling.state["bb_ca"][0, conditioning_only] == 0).all()
    assert held.coordinates.shape == (1, tokens, 37, 3)
    # Without the mask the same token is drawn and integrated like any other, so the check above can fail.
    assert (seen[0]["bb_ca"][0, conditioning_only] != 0).any()
    assert (free.sampling.state["bb_ca"][0, conditioning_only] != 0).any()


def test_rfproteina_native_builder_hoists_pair_condition_and_uses_bfloat16() -> None:
    trunk = _small_trunk_config().model_copy(update={"dtype": "bfloat16"})
    model = RFProteina(
        RFProteinaConfig(
            dtype="bfloat16",
            trunk=trunk,
            representation=RFProteinaRepresentationConfig(
                latent_dim=trunk.latent_dim,
                dtype="float32",
                pair_adaln_dtype="bfloat16",
            ),
            decoder=_small_decoder_config(),
        )
    ).cuda()
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, False, False, False],
        ],
        device="cuda",
    )
    batch, tokens = mask.shape
    state = {
        "bb_ca": torch.randn(batch, tokens, 3, device="cuda"),
        "local_latents": torch.randn(batch, tokens, 4, device="cuda"),
    }
    times = {
        "bb_ca": torch.tensor([0.25, 0.75], device="cuda"),
        "local_latents": torch.tensor([0.5, 0.125], device="cuda"),
    }
    self_conditioning = {name: torch.zeros_like(value) for name, value in state.items()}

    projection_input_shapes: list[torch.Size] = []
    adaln_conditions: list[torch.Tensor] = []

    def capture_projection_input(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        projection_input_shapes.append(args[0].shape)

    def capture_adaln_condition(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        adaln_conditions.append(args[1].detach().clone())

    handles = [
        model.representation_builder.pair_condition_projection.register_forward_pre_hook(capture_projection_input),
        model.representation_builder.pair_adaln.register_forward_pre_hook(capture_adaln_condition),
    ]
    try:
        representations = model.representation_builder(
            _native_static_features(mask),
            state,
            times,
            self_conditioning,
            False,
            mask,
        )
    finally:
        for handle in handles:
            handle.remove()

    pair_time = torch.cat(
        [
            representation_builder_module._time_embedding(
                times["bb_ca"], model.config.representation.time_embedding_dim
            ),
            representation_builder_module._time_embedding(
                times["local_latents"], model.config.representation.time_embedding_dim
            ),
        ],
        dim=-1,
    ).to(model.representation_builder.dtype)
    # The pair AdaLN takes one condition row per design and broadcasts it over the pairs.
    expected_condition = model.representation_builder.pair_condition_norm(
        model.representation_builder.pair_condition_projection(pair_time)
    ).to(model.representation_builder.pair_adaln_dtype)

    assert model.representation_builder.single_projection.weight.dtype == (torch.float32)
    assert model.representation_builder.pair_projection.weight.dtype == (torch.float32)
    assert model.representation_builder.pair_adaln.norm_cond.weight.dtype == (torch.bfloat16)
    assert projection_input_shapes == [torch.Size((batch, model.config.representation.pair_condition_input_dim))]
    assert len(adaln_conditions) == 1
    torch.testing.assert_close(adaln_conditions[0], expected_condition, atol=0, rtol=0)
    assert representations.single.dtype == torch.bfloat16
    assert representations.pair.dtype == torch.bfloat16
    assert representations.condition.dtype == torch.bfloat16


def test_decode_narrows_its_own_mask_without_touching_the_caller() -> None:
    """``decode`` excludes atomized tokens from encoding; the caller keeps its padding mask.

    A bool mask is its own ``.to(torch.bool)``, so narrowing one in place writes
    through to whatever the caller passed.
    """
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    ).cuda()
    init_module_weights(model.decoder)
    padding_mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")
    token_type = torch.tensor([[0, ATOMIZED_TOKEN_TYPE, 0, ATOMIZED_TOKEN_TYPE]], device="cuda")
    state = {
        "bb_ca": torch.zeros(1, 4, 3, device="cuda"),
        "local_latents": torch.zeros(1, 4, model.config.trunk.latent_dim, device="cuda"),
    }
    with torch.inference_mode():
        model.decode(state, padding_mask, token_type=token_type)

    assert padding_mask.all(), "decode narrowed the caller's padding mask in place"


def test_reconfiguring_graphs_repolicies_the_region_without_stacking_state() -> None:
    """Reconfiguring re-policies the one region and drops what the old policy captured.

    The region captures the trunk's own loop in place, so the trunk keeps its identity
    and its parameter paths, and a second ``configure_graphs`` cannot stack state on
    the first the way re-wrapping the module would.
    """
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            sampling=_deterministic_sampling_config(num_steps=1),
        )
    )
    trunk = model.velocity_model
    model.configure_graphs(capacity=2)
    model.configure_graphs(capacity=4)

    assert model.velocity_model is trunk, "the trunk module is never replaced"
    assert model.graph_status()["capacity"] == 4
    # The previous policy's tracker is gone; the next call builds one under the new policy.
    assert trunk.trunk_graph.tracker is None
    model.reset_graphs()
    assert model.graph_status()["enabled"] is False
    assert model.velocity_model is trunk


def test_released_recycle_schedules_match_the_upstream_strategies() -> None:
    """The shipped bands are the released ``recycle_strategy`` configs, and each sums to 60 steps.

    They are quoted here rather than imported because upstream is not a dependency; this is
    what pins them to it.
    """
    upstream = {
        "thinking_low": ([[5, 1], [5, 1], [5, 2], [5, 3], [10, 3], [5, 3], [5, 2], [10, 1], [10, 1]], 110),
        "thinking_medium": ([[5, 1], [5, 2], [5, 3], [5, 5], [10, 6], [5, 5], [5, 3], [10, 2], [10, 1]], 185),
        "thinking_high": ([[5, 2], [5, 3], [5, 5], [5, 8], [10, 9], [5, 8], [5, 5], [10, 3], [10, 2]], 295),
    }
    assert set(RECYCLE_SCHEDULES) == set(upstream)
    for name, (bands, passes) in upstream.items():
        shipped = recycle_schedule(name)
        assert shipped == [tuple(band) for band in bands], name
        assert sum(width for width, _ in shipped) == 60, f"{name} must cover the 60-step trajectory"
        assert sum(width * depth for width, depth in shipped) == passes, name
        # A sampling config built from the bands validates against that step count.
        RFProteinaSamplingConfig(num_steps=60, n_recycle_schedule=shipped)

    with pytest.raises(KeyError, match="unknown recycle schedule"):
        recycle_schedule("thinking_enormous")


@pytest.mark.parametrize("batch", [1, 2])
def test_reference_rng_takes_the_encoder_draw_and_nothing_else(batch: int) -> None:
    """``match_reference_rng`` must insert exactly one draw, ahead of x_0.

    The reference implementation encodes its input before drawing x_0, and the
    encoder's posterior sample takes one ``[B, N, latent_dim]`` draw from the
    stream. Matching it is what makes a seeded design reproduce the reference's
    noise; inserting anything more, or putting it anywhere else, shifts every
    later draw just as omitting it did.
    """
    latent_dim, tokens = 4, 5

    def run(match: bool) -> list[tuple[int, ...]]:
        shapes: list[tuple[int, ...]] = []

        def draw(shape, *, device, dtype):
            shapes.append(tuple(shape))
            return torch.zeros(shape, device=device, dtype=dtype)

        config = RFProteinaSamplingConfig(num_steps=3, match_reference_rng=match)
        sampler = RFProteinaProductSpaceSampler(config, latent_dim=latent_dim)
        mask = torch.ones(batch, tokens, device="cuda", dtype=torch.bool)

        def predict(state, times, self_conditioning, active, recycle_depth):
            del times, self_conditioning, active, recycle_depth
            return {modality: {"v": torch.zeros_like(value)} for modality, value in state.items()}

        with normal_draws(draw):
            sampler.sample(predict, mask, torch.float32, seed=11)
        return shapes

    production, reference = run(False), run(True)
    assert reference[0] == (batch, tokens, latent_dim)
    assert reference[1:] == production


def test_vf_tsr_scaled_score_drift_matches_the_rfd4_update() -> None:
    from bionemo_ir._torch.modules.rfproteina.schedules import compute_tsr_ratio
    from bionemo_ir._torch.modules.rfproteina.vf_tsr import score_to_velocity, velocity_to_score
    from bionemo_ir._torch.sampling.random import randn
    from bionemo_ir.models.rfproteina.config import RFProteinaVFStepConfig

    config = RFProteinaVFStepConfig(
        sampling_mode="vf_tsr", noise_scale=0.7, tsr_k=16.0, tsr_sigma=3.0, tsr_scaled_score_drift=True
    )
    torch.manual_seed(3)
    state, velocity = torch.randn(1, 6, 3), torch.randn(1, 6, 3)
    mask = torch.ones(1, 6, dtype=torch.bool)
    time, delta_time, noise_rate = torch.tensor([0.5]), torch.tensor([0.01]), torch.tensor([2.0])

    actual = velocity_flow_step(
        state, velocity, time, delta_time, noise_rate, mask, config,
        generator=torch.Generator().manual_seed(11), time_value=0.5,
    )  # fmt: skip

    score = velocity_to_score(state, velocity, time)
    scaled_score = score * compute_tsr_ratio(time, config.tsr_k, config.tsr_sigma)[..., None, None]
    noise = randn(state.shape, dtype=state.dtype, device=state.device, generator=torch.Generator().manual_seed(11))
    std = torch.sqrt(2.0 * noise_rate * config.noise_scale * delta_time)
    expected = (
        state + (score_to_velocity(state, scaled_score, time) + noise_rate * scaled_score) * delta_time + std * noise
    )
    torch.testing.assert_close(actual, expected)


def test_vf_tsr_scaled_score_drift_excludes_matched_noise() -> None:
    from bionemo_ir.models.rfproteina.config import RFProteinaVFStepConfig

    with pytest.raises(ValidationError, match="mutually exclusive"):
        RFProteinaVFStepConfig(tsr_scaled_score_drift=True, tsr_matched_noise=True)


def _vf_tsr_modality(**step: object):
    from bionemo_ir.models.rfproteina.config import RFProteinaModalitySamplingConfig

    return RFProteinaModalitySamplingConfig(
        schedule={"mode": "log", "power": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
        noise_schedule={"mode": "1/t", "power": 1.0},
        step={"sampling_mode": "vf_tsr", "ode_upper": 0.95, "tsr_k": 16.0, "tsr_sigma": 3.0, **step},
    )


def test_vf_tsr_scaled_score_drift_schedule_divides_the_noise_rate_by_the_ratio_squared() -> None:
    """RFD4 divides g(t) by the TSR ratio squared when its drift uses the TSR-scaled score; the other laws divide once."""
    from bionemo_ir._torch.modules.rfproteina.schedules import (
        compute_tsr_ratio,
        create_modality_schedule,
        get_noise_rate,
    )

    device = torch.device("cpu")
    scaled = create_modality_schedule(_vf_tsr_modality(tsr_scaled_score_drift=True), 60, device)
    once = create_modality_schedule(_vf_tsr_modality(), 60, device)
    times = scaled.times[:-1]
    ratio = compute_tsr_ratio(times, 16.0, 3.0)
    base = get_noise_rate(times, "1/t", 1.0, None)
    torch.testing.assert_close(scaled.noise_rate, base / ratio.square())
    torch.testing.assert_close(once.noise_rate, base / ratio)
    assert ratio.max() > 10.0, "a ratio near one would not tell the two divisions apart"


def test_vf_tsr_scaled_score_drift_is_the_old_matched_noise_update() -> None:
    """With score_scale 1, matched noise and the TSR-scaled drift are the same stochastic update, step for step."""
    from bionemo_ir._torch.modules.rfproteina.schedules import compute_tsr_ratio, create_modality_schedule

    device = torch.device("cpu")
    new = create_modality_schedule(_vf_tsr_modality(noise_scale=1.0, tsr_scaled_score_drift=True), 60, device)
    old = create_modality_schedule(_vf_tsr_modality(tsr_matched_noise=True), 60, device)
    torch.testing.assert_close(new.times, old.times)
    torch.manual_seed(5)
    state, velocity = torch.randn(1, 6, 3), torch.randn(1, 6, 3)
    mask = torch.ones(1, 6, dtype=torch.bool)
    new_step = _vf_tsr_modality(noise_scale=1.0, tsr_scaled_score_drift=True).step
    old_step = _vf_tsr_modality(tsr_matched_noise=True).step
    inside = [i for i, value in enumerate(new.times[:-1].tolist()) if 0.1 < value < 0.9]
    for index in inside[:: max(1, len(inside) // 3)][:3]:
        time = new.times[index].reshape(1)
        delta_time = (new.times[index + 1] - new.times[index]).reshape(1)
        assert 0.02 < float(time) < 0.95 and compute_tsr_ratio(time, 16.0, 3.0) > 1.0
        outputs = [
            velocity_flow_step(
                state,
                velocity,
                time,
                delta_time,
                schedule.noise_rate[index].reshape(1),
                mask,
                step,
                generator=torch.Generator().manual_seed(9),
                time_value=float(time),
            )  # fmt: skip
            for schedule, step in ((new, new_step), (old, old_step))
        ]
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
