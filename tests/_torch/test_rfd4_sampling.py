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

import bionemo_ir._torch.modules.rfproteina.representation_builder as representation_builder_module
from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.layers.triangle_nodes import TriangleMultiplicationNode
from bionemo_ir._torch.modules.rfproteina.contracts import RFProteinaSimulationOutput
from bionemo_ir._torch.modules.rfproteina.flow_matcher import RFProteinaProductSpaceFlowMatcher
from bionemo_ir._torch.modules.rfproteina.groups import GROUP_KEYS, extract_groups
from bionemo_ir._torch.modules.rfproteina.simulator import RFProteinaProductSpaceSimulator
from bionemo_ir._torch.modules.rfproteina.vf_tsr import velocity_flow_step, velocity_to_clean
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.flow_matching import (
    FlowMatchingIntegrator,
    FlowMatchingIntegratorConfig,
    FlowMatchingRolloutPlan,
    FlowMatchingSchedule,
)
from bionemo_ir._torch.sampling.runner import GenerativeRunner, SamplingCancelled
from bionemo_ir.configs import BaseConfig
from bionemo_ir.hubs import resolve_rfproteina_autoencoder_checkpoint, resolve_rfproteina_checkpoint
from bionemo_ir.models.optimize_module_setter import AcceleratedConfig, BackendType
from examples.design.rfproteina import load_model
from tests._torch.rfproteina_utils import get_residue_atom_mask

# isort: off
from bionemo_ir._torch.modules.rfproteina.representation_builder import ATOMIZED_TOKEN_TYPE
from bionemo_ir.models.rfproteina.config import (
    RFProteinaDecoderConfig,
    RFProteinaEncoderConfig,
    RFProteinaReferenceConfig,
    RFProteinaRepresentationConfig,
    RFProteinaTrunkConfig,
)
from bionemo_ir.models.rfproteina import (
    RFProteina,
    RFProteinaConfig,
    RFProteinaSimulationConfig,
)
from bionemo_ir._torch.modules.rfproteina.contracts import RFProteinaRepresentations
from bionemo_ir.models.rfproteina.convert import (
    load_rfproteina_checkpoint_state,
)

# isort: on
from tests._torch import init_module_weights


def _run_model(model, mask, representation_builder=None, *, static_features=None, seed=0, num_steps=4):
    from contextlib import nullcontext
    from unittest.mock import patch

    features = {"padding_mask": mask} if static_features is None else static_features
    generated = mask & features.get("flow_backbone_mask", mask)
    grids = {
        "ts": {key: torch.linspace(0.1, 1.0, num_steps + 1, device=mask.device) for key in ("bb_ca", "local_latents")},
        "gt": {key: torch.zeros(num_steps, device=mask.device) for key in ("bb_ca", "local_latents")},
        "recycle_depths": [1] * num_steps,
    }

    def predict(state, times, clean, active, recycle_depth):
        return model(representation_builder(state, times, clean, active), mask, recycle_depth=recycle_depth)

    override = (
        nullcontext()
        if representation_builder is None
        else patch.object(model, "_make_predict_fn", return_value=predict)
    )
    with SamplingContext.resolve(mask.device, seed=seed) as context, override:
        state = RFProteinaProductSpaceFlowMatcher(model.config.simulation, model.config.trunk.latent_dim).sample_noise(
            generated, generator=context.generator
        )
        return model.partial_simulation(features, state, start_step=0, end_step=num_steps, context=context, **grids)


def _deterministic_simulation_config() -> RFProteinaSimulationConfig:
    config = RFProteinaSimulationConfig(self_condition=True)
    config.bb_ca.step.sampling_mode = "vf"
    config.local_latents.step.sampling_mode = "vf"
    config.bb_ca.reference.stochastic_centering_mode = "none"
    return config


def test_rfproteina_velocity_to_clean() -> None:
    state = torch.tensor([[[1.0, 2.0]]])
    velocity = torch.tensor([[[2.0, -2.0]]])
    time = torch.tensor([0.25])
    mask = torch.tensor([[True]])
    clean = velocity_to_clean(state, velocity, time, mask)
    torch.testing.assert_close(clean, torch.tensor([[[2.5, 0.5]]]))


def test_rfproteina_product_sampler_self_conditioning() -> None:
    config = _deterministic_simulation_config()
    sampler = RFProteinaProductSpaceSimulator(config, latent_dim=4)
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

    output = sampler.partial_simulation(
        predict,
        mask,
        torch.float32,
        initial_state=initial,
        schedules=_linear_plan(4, ("bb_ca", "local_latents")).schedule,
        recycle_depths=[1] * 4,
        carry_self_condition=False,
        save_trajectory_every=2,
    )
    torch.testing.assert_close(output.state["bb_ca"], initial["bb_ca"])
    torch.testing.assert_close(output.state["local_latents"], initial["local_latents"])
    assert self_condition_flags == [False, True, True, True]
    assert len(output.trajectory) == 3


def test_simulator_uses_the_callers_recycle_depths() -> None:
    simulator = RFProteinaProductSpaceSimulator(_deterministic_simulation_config(), latent_dim=4)
    mask = torch.ones(1, 2, device="cuda", dtype=torch.bool)
    state = {key: torch.zeros(1, 2, width, device="cuda") for key, width in (("bb_ca", 3), ("local_latents", 4))}
    depths = []

    def predict(state, times, clean, active, recycle_depth):
        depths.append(recycle_depth)
        return {key: {"v": torch.zeros_like(value)} for key, value in state.items()}

    simulator.partial_simulation(
        predict,
        mask,
        torch.float32,
        state,
        schedules=_linear_plan(4, ("bb_ca", "local_latents")).schedule,
        recycle_depths=[1, 2, 3, 3],
        carry_self_condition=False,
        seed=11,
    )
    assert depths == [1, 2, 3, 3]


def test_rfproteina_inverse_folding_resamples_backbone_and_restores_target() -> None:
    config = _deterministic_simulation_config()
    sampler = RFProteinaProductSpaceSimulator(config, latent_dim=4)
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

    output = sampler.partial_simulation(
        predict,
        mask,
        torch.float32,
        clean,
        schedules=_linear_plan(3, ("bb_ca", "local_latents")).schedule,
        recycle_depths=[1] * 3,
        carry_self_condition=False,
        target_backbone=target,
        seed=7,
    )
    torch.testing.assert_close(output.state["bb_ca"], target)
    assert not torch.equal(output.state["local_latents"], clean["local_latents"])
    assert len(seen_state_backbone) == 3
    # First step is a forward-marginal draw, not the model's previous backbone update.
    assert not torch.allclose(seen_state_backbone[0], torch.full_like(target, 99.0))
    torch.testing.assert_close(seen_self_cond[1], target)
    torch.testing.assert_close(seen_self_cond[2], target)


def _linear_plan(num_steps: int, modalities=("x",)) -> FlowMatchingRolloutPlan:
    times = torch.linspace(0.0, 1.0, num_steps + 1, device="cuda")
    return FlowMatchingRolloutPlan(
        schedule={key: FlowMatchingSchedule(times=times) for key in modalities},
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
    num_steps = 5
    config = RFProteinaSimulationConfig()
    for modality in (config.bb_ca, config.local_latents):
        modality.step.sampling_mode = sampling_mode

    latent_dim = 4
    mask = torch.tensor([[True] * 6 + [False] * 2], device="cuda")
    can_be_encoded = torch.tensor([[True] * 5 + [False] * 3], device="cuda")
    torch.manual_seed(7)
    velocities = [
        {
            "bb_ca": torch.randn(1, 8, 3, device="cuda"),
            "local_latents": torch.randn(1, 8, latent_dim, device="cuda"),
        }
        for _ in range(num_steps)
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

    sampler = RFProteinaProductSpaceSimulator(config, latent_dim=latent_dim)
    initial = {key: torch.zeros_like(value) for key, value in velocities[0].items()}
    schedules = {
        key: FlowMatchingSchedule(
            times=torch.linspace(0.1, 1.0, num_steps + 1, device="cuda"),
            noise_rate=torch.ones(num_steps, device="cuda"),
        )
        for key in initial
    }
    output = sampler.partial_simulation(
        predict,
        mask,
        torch.float32,
        initial,
        can_be_encoded=can_be_encoded,
        schedules=schedules,
        recycle_depths=[1] * num_steps,
        carry_self_condition=False,
        seed=1234,
        save_trajectory_every=2,
    )
    reference_generator = torch.Generator(device="cuda").manual_seed(1234)
    state = {key: value.clone() for key, value in initial.items()}
    modality_configs = {"bb_ca": config.bb_ca, "local_latents": config.local_latents}
    clean = {key: torch.zeros_like(value) for key, value in state.items()}
    trajectory_length = 0
    for step_index in range(num_steps):
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
        if step_index % 2 == 0 or step_index == num_steps - 1:
            trajectory_length += 1

    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(output.state[modality], state[modality])
        torch.testing.assert_close(output.clean_prediction[modality], clean[modality])
    assert len(output.trajectory) == trajectory_length
    assert output.seed == 1234
    assert predict.calls == num_steps


def _centering_config(scope: str, group_keys: tuple[str, ...] = ()) -> RFProteinaSimulationConfig:
    config = RFProteinaSimulationConfig()
    reference = config.bb_ca.reference
    reference.stochastic_centering_scope = scope
    reference.stochastic_centering_group_keys = group_keys
    return config


def _draw_backbone(config: RFProteinaSimulationConfig, padding_mask, **kwargs):
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


def test_simulator_seed_is_reproducible_and_reported() -> None:
    config = RFProteinaSimulationConfig()
    simulator = RFProteinaProductSpaceSimulator(config, latent_dim=4)
    mask = torch.ones(1, 6, dtype=torch.bool, device="cuda")
    state = {key: torch.ones(1, 6, width, device="cuda") for key, width in (("bb_ca", 3), ("local_latents", 4))}

    def predict(state, times, clean, active, recycle_depth):
        return {key: {"v": torch.zeros_like(value)} for key, value in state.items()}

    schedules = {
        key: FlowMatchingSchedule(
            times=torch.linspace(0.1, 1.0, 4, device="cuda"), noise_rate=torch.ones(3, device="cuda")
        )
        for key in state
    }

    def run(seed):
        return simulator.partial_simulation(
            predict,
            mask,
            torch.float32,
            state,
            schedules=schedules,
            recycle_depths=[1] * 3,
            carry_self_condition=False,
            seed=seed,
        )

    first, second, other = run(99), run(99), run(100)
    assert first.seed == 99
    for key in state:
        torch.testing.assert_close(first.state[key], second.state[key])
        assert not torch.allclose(first.state[key], other.state[key])


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


def _small_model() -> RFProteina:
    """The small float32 model with the deterministic simulation config that most model-level tests build."""
    return RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=_small_trunk_config(),
            decoder=_small_decoder_config(),
            simulation=_deterministic_simulation_config(),
        )
    )


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


class _FixedHead(torch.nn.Module):
    """An output head that returns chosen values, whatever the features are."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        return self.value.to(single.dtype)


class _DecoderPredictions(torch.nn.Module):
    """Fixed numerical predictions for layout tests."""

    num_residue_types = 33
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()

    def forward(self, backbone_ca, local_latents, mask, **kwargs):
        del backbone_ca, local_latents, kwargs
        return {
            "coords": torch.zeros(*mask.shape, 37, 3),
            "sequence_logits": torch.zeros(*mask.shape, 33),
            "residue_type": torch.ones_like(mask, dtype=torch.long) * mask,
        }


def _model_with_decoder(decoder: torch.nn.Module) -> RFProteina:
    model = _small_model()
    model.decoder = decoder
    return model


@pytest.mark.parametrize("sliced", [False, True])
def test_decode_returns_presence_logits_in_original_token_layout(sliced: bool) -> None:
    class PresenceDecoder(_DecoderPredictions):
        def forward(self, backbone_ca, local_latents, mask, **kwargs):
            result = super().forward(backbone_ca, local_latents, mask, **kwargs)
            result["atom_presence_logits"] = backbone_ca[..., :1].expand(*mask.shape, 37).clone()
            return result

    model = _model_with_decoder(PresenceDecoder())
    model.config.decoder.use_atom_presence = True
    padding = torch.tensor([[True, True, True], [True, True, False], [True, True, True]])
    emit = torch.tensor([[True, False, True], [False, False, False], [False, True, False]]) if sliced else padding
    state = {"bb_ca": torch.arange(27).reshape(3, 3, 3).float(), "local_latents": torch.zeros(3, 3, 4)}
    decoded = model.decode(state, padding, can_be_encoded=emit)
    expected = state["bb_ca"][..., :1].expand(3, 3, 37)
    torch.testing.assert_close(decoded["atom_presence_logits"][emit], expected[emit])
    assert torch.isneginf(decoded["atom_presence_logits"][~emit]).all()


_ALA, _GLY, _TYR = 1, 8, 19
_BACKBONE = [0, 1, 2, 4]  # N, CA, C, O


def _encoder_setup(chemistry: bool = False) -> tuple[RFProteina, dict[str, torch.Tensor]]:
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
                simulation=_deterministic_simulation_config(),
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


def test_rfproteina_encode_reads_the_chemistry_of_each_atom() -> None:
    model, static_features = _encoder_setup(chemistry=True)
    occupied = static_features["atom37_mask"]
    carbon = torch.where(occupied, 6, 0)
    nitrogen = carbon.clone()
    nitrogen[0, 1] = torch.where(occupied[0, 1], 7, 0)
    bonds = torch.zeros(1, 4, 37, 10, device="cuda")

    def encode(**chemistry):
        features = {**static_features, **chemistry}
        return model.encode(features, sample_latent=False).mean

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
            simulation=_deterministic_simulation_config(),
        )
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

    handle = model.representation_builder.single_projection.register_forward_pre_hook(capture("single"))
    model.representation_builder(
        static_features,
        state,
        times,
        self_conditioning,
        False,
        static_features["padding_mask"],
    )
    handle.remove()
    # The pair kernel reads the nucleic-acid pair channels as the last two prepared scalars.
    projection_inputs["pair"] = model.representation_builder.prepare(static_features).pair_features.scalars

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


def test_rfproteina_velocity_trunk_cudagraph_matches_eager() -> None:
    """Graphing the velocity trunk must not change the rollout.

    The trunk sees identical shapes on every sampling step, so one capture
    serves the whole rollout. The run must be long enough to get past the
    tracker's warmup thresholds, otherwise this would silently only exercise
    the eager fallback -- hence the explicit capture assertion.
    """
    trunk = _small_trunk_config()
    sampling = _deterministic_simulation_config()
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
                RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config(), simulation=sampling)
            )
            .cuda()
            .eval()
        )
        init_module_weights(model)
        return model

    eager = build_model()
    eager.reset_graphs()
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

    reference = _run_model(eager, mask, build_representations, seed=5)
    captured = _run_model(graphed, mask, build_representations, seed=5)

    tracker = graphed.velocity_model.trunk_graph.tracker
    assert isinstance(tracker, CUDAGraphOptimizationTracker)
    states = list(tracker.graph_state_by_key.values())
    assert len(states) == 1, "one shape across the rollout means one graph, whatever its depths"
    assert states[0].preparation_state == CUDAGraphPreparationState.GRAPH_VERIFIED
    assert not tracker.fallback_to_eager_by_key, "captured, not run eagerly"

    for modality in ("bb_ca", "local_latents"):
        torch.testing.assert_close(captured.state[modality], reference.state[modality], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("loader", ["state_dicts", "example"])
def test_rfproteina_default_graphs_match_eager_with_released_weights(loader: str) -> None:
    """The default loader captures a real trunk and preserves seeded designs bit for bit."""
    flow = resolve_rfproteina_checkpoint()
    autoencoder = resolve_rfproteina_autoencoder_checkpoint()
    if flow is None or not flow.is_file() or autoencoder is None or not autoencoder.is_file():
        pytest.skip("RFPROTEINA_LDM_CKPT and RFPROTEINA_AE_CKPT are not set to the released checkpoints")
    if loader == "example":
        model = load_model(flow, autoencoder)
    else:
        model = RFProteina.from_reference_state_dicts(
            load_rfproteina_checkpoint_state(flow),
            load_rfproteina_checkpoint_state(autoencoder),
            simulation_config=RFProteinaSimulationConfig(),
            device="cuda",
        )
    assert model.graph_status()["enabled"]
    features = _native_static_features(torch.ones(1, 16, dtype=torch.bool, device="cuda"))
    for token in representation_builder_module.SEQUENCE_PROBABILITY_TOKENS:
        features[f"condition_sequence-probability-{token}_1_residue"] = torch.full((1, 16), torch.nan, device="cuda")
    with torch.inference_mode():
        first = _run_model(model, features["padding_mask"], static_features=features, seed=42)
        replay = _run_model(model, features["padding_mask"], static_features=features, seed=42)
        status = model.graph_status()
        assert status["captured"] == 1, status
        assert status["fallback"] == 0, status
        model.reset_graphs()
        eager = _run_model(model, features["padding_mask"], static_features=features, seed=42)
    for output in (first, replay):
        for modality in ("bb_ca", "local_latents"):
            assert torch.equal(output.state[modality], eager.state[modality]), modality
        decoded = model.decode(output.state, features["padding_mask"])
        reference = model.decode(eager.state, features["padding_mask"])
        for name in decoded:
            assert torch.equal(decoded[name], reference[name]), name


def test_rfproteina_trunk_graph_default_and_opt_out() -> None:
    """Each config owns one bounded cache and callers can choose eager execution."""
    config = _small_trunk_config()
    policy = config.graph_optimization_config
    assert policy is not None
    # A raw-feature caller can request the final pair separately from recycled passes.
    assert policy.num_graphs_max_for_this_module == 2
    assert policy.input_routing_config.input_acceptance_dims[0].dim_len_max == 511
    assert _small_trunk_config().graph_optimization_config is not policy
    model = RFProteina(
        RFProteinaConfig(
            dtype="float32",
            trunk=config.model_copy(update={"graph_optimization_config": None}),
            decoder=_small_decoder_config(),
        )
    )
    assert model.graph_status()["enabled"] is False


@pytest.mark.parametrize(("tokens", "captures"), [(504, 1), (505, 0)])
def test_rfproteina_default_graph_cap_applies_after_token_padding(tokens: int, captures: int) -> None:
    """Alignment to 512 tokens must take the eager path under the default cap."""
    trunk = _small_trunk_config()
    model = RFProteina(RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config())).cuda().eval()
    init_module_weights(model)
    with torch.inference_mode():
        model.velocity_model(
            single=torch.randn(1, tokens, trunk.token_dim, device="cuda"),
            pair=torch.randn(1, tokens, tokens, trunk.pair_dim, device="cuda"),
            condition=torch.randn(1, tokens, trunk.condition_dim, device="cuda"),
            padding_mask=torch.ones(1, tokens, dtype=torch.bool, device="cuda"),
        )
    status = model.graph_status()
    assert status["captured"] == captures, status
    assert status["fallback"] == 0, status


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
    sampling = _deterministic_simulation_config()
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
                RFProteinaConfig(dtype="bfloat16", trunk=trunk, decoder=_small_decoder_config(), simulation=sampling)
            )
            .cuda()
            .eval()
        )
        init_module_weights(model)
        return model

    eager = build_model()
    eager.reset_graphs()
    if not any(isinstance(module, TriangleMultiplicationNode) and module.kf_available for module in eager.modules()):
        pytest.skip("the fused TriMul KF chain does not ship for this GPU or these widths")

    graphed = build_model()
    graphed.configure_graphs(capacity=4, verify_capture=True)

    reference = _run_model(eager, mask, build_representations, seed=5)
    first = _run_model(graphed, mask, build_representations, seed=5)
    second = _run_model(graphed, mask, build_representations, seed=5)

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
    sampling = _deterministic_simulation_config()
    model = RFProteina(
        RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config(), simulation=sampling)
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

    output = _run_model(model, mask, build_representations)
    assert output.state["bb_ca"].shape == (1, 10, 3)
    assert output.state["local_latents"].shape == (1, 10, 4)
    assert torch.isfinite(output.state["bb_ca"]).all()
    assert torch.isfinite(output.state["local_latents"]).all()
    decoded = model.decode(output.state, mask)
    assert decoded["coords"].shape == (1, 10, 37, 3)
    assert decoded["sequence_logits"].shape == (1, 10, 20)
    assert decoded["residue_type"].shape == (1, 10)
    assert active_flags == [False, True, True, True]


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
            simulation=_deterministic_simulation_config(),
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

    output = _run_model(model, mask, static_features=static_features, seed=17)
    assert output.seed == 17
    assert output.insertion_logits is not None
    assert output.insertion_logits.shape == (1, 10, 10)
    assert torch.isfinite(output.insertion_logits).all()
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    assert torch.count_nonzero(output.insertion_logits[~pair_mask]) == 0
    decoded = model.decode(output.state, mask)
    assert decoded["coords"].shape == (1, 10, 37, 3)
    assert decoded["sequence_logits"].shape == (1, 10, 20)
    assert torch.isfinite(decoded["coords"]).all()
    assert torch.isfinite(decoded["sequence_logits"]).all()


def test_rfproteina_resolves_motif_runtime_masks_from_static_features() -> None:
    model = _small_model()
    mask = torch.tensor([[True, True, True, True, False]])
    static_features = _native_static_features(mask)
    static_features["can_be_encoded"] = torch.tensor([[True, False, True, True, True]])
    static_features["has_known_index"] = torch.tensor([[True, True, False, True, True]])
    coordinate_mask = torch.zeros(1, 5, 37, dtype=torch.bool)
    coordinate_mask[0, 1, 36] = True
    coordinate_mask[0, 4, 0] = True
    static_features["mask_coordinate_1_atom"] = coordinate_mask

    recorder = _MaskRecordingSampler()
    model.simulator = recorder
    _run_model(model, mask, static_features=static_features)

    torch.testing.assert_close(
        recorder.can_be_encoded,
        torch.tensor([[True, False, False, True, False]]),
    )
    torch.testing.assert_close(
        recorder.coordinate_condition_mask,
        torch.tensor([[False, True, False, False, False]]),
    )

    static_features["mask_coordinate_1_atom"] = torch.zeros(1, 5, 36, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask_coordinate_1_atom"):
        _run_model(model, mask, static_features=static_features)


class _MaskRecordingSampler:
    """Stands in for the sampler and keeps the masks the model hands it."""

    group_keys = frozenset()

    def __init__(self) -> None:
        self.padding_mask = None
        self.can_be_encoded = None
        self.coordinate_condition_mask = None

    def partial_simulation(self, predict, padding_mask, dtype, **kwargs):
        del predict, dtype
        self.padding_mask = padding_mask
        self.can_be_encoded = kwargs["can_be_encoded"]
        self.coordinate_condition_mask = kwargs["coordinate_condition_mask"]
        state = {"bb_ca": torch.zeros(*padding_mask.shape, 3), "local_latents": torch.zeros(*padding_mask.shape, 4)}
        return RFProteinaSimulationOutput(
            state=state, clean_prediction=state, insertion_logits=None, trajectory=[], seed=kwargs["seed"]
        )


def test_rfproteina_flow_backbone_mask_is_the_rollout_mask_and_narrows_latents() -> None:
    """A conditioning-only token stays a real token for the trunk but leaves the rollout.

    The sampler takes its generated-token mask where it takes the padding mask, and latents are emitted
    only where the backbone is, so a caller that supplies one mask cannot leave the other disagreeing.
    """
    model = _small_model()
    mask = torch.tensor([[True, True, True, True, False]])
    static_features = _native_static_features(mask)
    static_features["flow_backbone_mask"] = torch.tensor([[True, False, True, True, True]])
    recorder = _MaskRecordingSampler()
    model.simulator = recorder

    _run_model(model, mask, static_features=static_features)

    # Token 1 only conditions; token 4 is padding, which the backbone mask cannot bring back.
    torch.testing.assert_close(recorder.padding_mask, torch.tensor([[True, False, True, True, False]]))
    torch.testing.assert_close(recorder.can_be_encoded, torch.tensor([[True, False, True, True, False]]))

    del static_features["flow_backbone_mask"]
    _run_model(model, mask, static_features=static_features)
    torch.testing.assert_close(recorder.padding_mask, mask)
    torch.testing.assert_close(recorder.can_be_encoded, mask)

    static_features["flow_backbone_mask"] = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="flow_backbone_mask"):
        model.partial_simulation(static_features, {}, ts={}, gt={}, recycle_depths=[])


class _EmitMaskRecordingHeads(torch.nn.Module):
    """A frontend readout that emits zero velocity and keeps the emit masks the model hands it."""

    requires_pair = False

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seen: list[dict[str, torch.Tensor]] = []

    def forward(self, single, pair, padding_mask, emit_masks=None):
        del pair, padding_mask
        self.seen.append({name: mask.clone() for name, mask in emit_masks.items()})
        batch, tokens = single.shape[:2]
        return {
            "bb_ca": {"v": single.new_zeros(batch, tokens, 3)},
            "local_latents": {"v": single.new_zeros(batch, tokens, self.latent_dim)},
        }


class _FixedFeatureBuilder(torch.nn.Module):
    """Hands the trunk fixed features, so a rollout reaches its readout without the native feature plumbing."""

    output_dtype = torch.float32
    pair_adaln_dtype = torch.bfloat16

    def __init__(self, representations: RFProteinaRepresentations) -> None:
        super().__init__()
        self.representations = representations

    def prepare(self, static_features, padding_mask):
        return object()

    def forward(self, **kwargs):
        return self.representations


def test_rfproteina_heads_get_the_masks_the_sampler_integrates_with() -> None:
    """A host that does not narrow its own masks still gets a readout that agrees with the rollout.

    The host marks a conditioning-only token, an atomized token and a token of unknown index as emitting
    latents, and marks a padded token as generated. The sampler drops each of them, so a readout that took the
    raw features would emit velocity for tokens the rollout holds at zero.
    """
    model = _small_model().cuda()
    init_module_weights(model)
    trunk, tokens = model.config.trunk, 6
    mask = torch.tensor([[True, True, True, True, True, False]], device="cuda")
    features = _native_static_features(mask)
    features["token_type"][0, 2] = ATOMIZED_TOKEN_TYPE
    features["flow_backbone_mask"] = torch.tensor([[True, False, True, True, True, True]], device="cuda")
    features["can_be_encoded"] = torch.ones(1, tokens, dtype=torch.bool, device="cuda")
    features["has_known_index"] = torch.tensor([[True, True, True, False, True, True]], device="cuda")
    heads = _EmitMaskRecordingHeads(trunk.latent_dim)
    model.velocity_model.heads = heads
    model.representation_builder = _FixedFeatureBuilder(
        RFProteinaRepresentations(
            single=torch.randn(1, tokens, trunk.token_dim, device="cuda"),
            pair=torch.randn(1, tokens, tokens, trunk.pair_dim, device="cuda"),
            condition=torch.randn(1, tokens, trunk.condition_dim, device="cuda"),
        )
    )
    sampler_masks: dict[str, torch.Tensor] = {}
    simulate = model.simulator.partial_simulation

    def spy(**kwargs):
        sampler_masks.update(generated=kwargs["padding_mask"], latents=kwargs["can_be_encoded"])
        return simulate(**kwargs)

    model.simulator.partial_simulation = spy
    _run_model(model, mask, static_features=features)

    generated = torch.tensor([[True, False, True, True, True, False]], device="cuda")
    latents = torch.tensor([[True, False, False, False, True, False]], device="cuda")
    assert heads.seen, "the rollout never reached the readout"
    for emit_masks in heads.seen:
        torch.testing.assert_close(emit_masks["bb_ca"], sampler_masks["generated"])
        torch.testing.assert_close(emit_masks["local_latents"], sampler_masks["latents"])
        torch.testing.assert_close(emit_masks["bb_ca"], generated)
        torch.testing.assert_close(emit_masks["local_latents"], latents)


def test_rfproteina_reference_state_centers_over_the_generated_tokens_only() -> None:

    config = RFProteinaSimulationConfig()
    config.bb_ca.reference.zero_com_noise = True
    config.bb_ca.reference.stochastic_centering_mode = "none"  # the offset would move the centre on purpose
    generated = torch.tensor([[True, False, True, True, True, True, False]], device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7)

    state = RFProteinaProductSpaceFlowMatcher(config, 4).sample_noise(generated, generator=generator)

    assert (state["bb_ca"][:, ~generated[0]] == 0).all(), "a conditioning-only token starts at the origin"
    torch.testing.assert_close(
        state["bb_ca"][0, generated[0]].mean(0), torch.zeros(3, device="cuda"), atol=1e-6, rtol=0
    )


def test_rfproteina_conditioning_only_tokens_hold_zero_state_through_a_rollout() -> None:
    trunk = _small_trunk_config()
    sampling = _deterministic_simulation_config()
    sampling.bb_ca.reference.zero_com_noise = True
    model = RFProteina(
        RFProteinaConfig(dtype="float32", trunk=trunk, decoder=_small_decoder_config(), simulation=sampling)
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

    held = _run_model(
        model, mask, build_representations, static_features={"padding_mask": mask, "flow_backbone_mask": generated}
    )
    held_states, seen[:] = list(seen), []
    free = _run_model(model, mask, build_representations, static_features={"padding_mask": mask})

    assert len(held_states) == 4
    for snapshot in held_states:
        # The trunk is handed every token on every step, and the held one never leaves the origin.
        assert snapshot["bb_ca"].shape == (1, tokens, 3)
        assert (snapshot["bb_ca"][0, conditioning_only] == 0).all()
        assert (snapshot["local_latents"][0, conditioning_only] == 0).all()
        assert (snapshot["bb_ca"][0, generated[0]] != 0).any()
    torch.testing.assert_close(
        held_states[0]["bb_ca"][0, generated[0]].mean(0), torch.zeros(3, device="cuda"), atol=1e-6, rtol=0
    )
    assert (held.state["bb_ca"][0, conditioning_only] == 0).all()
    assert model.decode(held.state, mask)["coords"].shape == (1, tokens, 37, 3)
    # Without the mask the same token is drawn and integrated like any other, so the check above can fail.
    assert (seen[0]["bb_ca"][0, conditioning_only] != 0).any()
    assert (free.state["bb_ca"][0, conditioning_only] != 0).any()


def _unified_builder_inputs(
    simulation: RFProteinaSimulationConfig | None = None, **representation: bool | str
) -> tuple[RFProteina, dict[str, torch.Tensor], dict, dict, dict, torch.Tensor]:
    """A recycling model whose builder has the unified checkpoints' sequence-probability schema and remaining-pass
    conditioning, and one step's inputs: two designs of different lengths, with populated coordinate groups.
    ``representation`` overrides builder settings, such as the chirality alignment."""
    trunk = _small_trunk_config().model_copy(update={"dtype": "bfloat16", "recycle_enabled": True})
    model = RFProteina(
        RFProteinaConfig(
            dtype="bfloat16",
            trunk=trunk,
            representation=RFProteinaRepresentationConfig(
                **{
                    "latent_dim": trunk.latent_dim,
                    "input_schema": "sequence_probability",
                    "recycle_conditioning": "recycle_remaining",
                    "dtype": "float32",
                    "pair_adaln_dtype": "bfloat16",
                    **representation,
                }
            ),
            decoder=_small_decoder_config(),
            **({} if simulation is None else {"simulation": simulation}),
        )
    ).cuda()
    mask = torch.tensor([[True] * 6 + [False] * 2, [True] * 5 + [False] * 3], device="cuda")
    batch, tokens = mask.shape
    static = _native_static_features(mask)
    for token in representation_builder_module.SEQUENCE_PROBABILITY_TOKENS:
        static[f"condition_sequence-probability-{token}_1_residue"] = torch.full(
            (batch, tokens), torch.nan, device="cuda"
        )
    static["condition_sequence-probability-ALA_1_residue"][:, :2] = 0.5
    groups = static["condition_CoordinateGroup_1_atom"]
    groups[:, 0, :3] = 2
    groups[:, 2, 1] = 5
    groups[1, 4, 0] = 0
    state = {
        "bb_ca": torch.randn(batch, tokens, 3, device="cuda"),
        "local_latents": torch.randn(batch, tokens, trunk.latent_dim, device="cuda"),
    }
    times = {
        "bb_ca": torch.tensor([0.25, 0.75], device="cuda"),
        "local_latents": torch.tensor([0.5, 0.125], device="cuda"),
    }
    self_conditioning = {name: torch.zeros_like(value) for name, value in state.items()}
    return model, static, state, times, self_conditioning, mask


def test_a_step_on_prepared_static_parts_matches_a_full_build_and_never_waits_on_the_device() -> None:
    """A design's static track parts make a step's tracks bit for bit what a build from the features alone makes.

    The step reads the coordinate groups' distinct labels from the parts, so it draws the same permutation and never
    synchronizes with the device; its condition stack, one per trunk pass, is the one built from the features.
    """
    model, static, state, times, self_conditioning, mask = _unified_builder_inputs()
    builder = model.representation_builder
    with torch.inference_mode():
        parts = builder.prepare(static, mask)
        torch.manual_seed(5)
        expected = builder(static, state, times, self_conditioning, False, mask)
        expected_stack = builder.condition_stack(static, times, mask, 3)
        torch.manual_seed(5)
        torch.cuda.set_sync_debug_mode("error")
        try:
            reused = builder(static, state, times, self_conditioning, False, mask, parts=parts, recycle_depth=3)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    assert parts.coordinate_groups.count == 3
    for track in ("single", "pair"):
        assert torch.equal(getattr(reused, track), getattr(expected, track)), track
    assert torch.equal(reused.condition, expected_stack)
    assert torch.equal(expected_stack[0], expected.condition)


@pytest.mark.parametrize("frames", [False, True], ids=["no-frames", "frames"])
def test_noise_aligned_chirality_finds_its_frames_once_per_rollout(frames: bool) -> None:
    """A step aligns chirality with the frames its rollout found once, bit for bit as a build from the features
    alone does. Without a frame the chirality is static, so the step never waits on the device."""
    model, static, state, times, self_conditioning, mask = _unified_builder_inputs(noise_align_chirality=True)
    batch, tokens = mask.shape
    slots = 6
    static["chirality"] = torch.randn(batch, tokens, 3 * slots, device="cuda")
    frame_atoms = torch.full((batch, tokens, slots, 8), -1, dtype=torch.long, device="cuda")
    owners = torch.full((batch, tokens, slots), -1, dtype=torch.long, device="cuda")
    if frames:
        # A stereocentre at token 0 whose neighbours, tokens 1 to 3, carry slot 0.
        frame_atoms[0, 0, 0, :5] = torch.tensor([0, 0, 1, 2, 3])
        owners[0, 1:4, 0] = 0
    static.update(chiral_frame_atoms=frame_atoms, chiral_slot_owner=owners)
    builder = model.representation_builder
    with torch.inference_mode():
        parts = builder.prepare(static, mask)
        torch.manual_seed(5)
        expected = builder(static, state, times, self_conditioning, False, mask)
        torch.manual_seed(5)
        torch.cuda.set_sync_debug_mode("default" if frames else "error")
        try:
            reused = builder(static, state, times, self_conditioning, False, mask, parts=parts)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    assert (parts.chirality is None) == frames
    assert parts.chiral_frames[0].numel() == int(frames)
    for track in ("single", "pair", "condition"):
        assert torch.equal(getattr(reused, track), getattr(expected, track)), track


def _one_hot_pair(
    builder: torch.nn.Module,
    static: dict[str, torch.Tensor],
    mask: torch.Tensor,
    coordinates: torch.Tensor,
    self_coordinates: torch.Tensor,
    active: float,
    time: torch.Tensor,
) -> torch.Tensor:
    """The upstream pair track in FP64: the concatenated one-hot and scalar pair features projected, normalized,
    rounded to the AdaLN dtype as the norm stores it, conditioned per design and masked."""
    config = builder.config
    legacy = config.input_schema == "legacy_self_conditioned"
    edges = torch.linspace(config.pair_distance_min, config.pair_distance_max, config.pair_distance_bins - 1).cuda()

    def bins(x: torch.Tensor) -> torch.Tensor:
        distance = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), dim=-1)
        return torch.nn.functional.one_hot(torch.bucketize(distance, edges), config.pair_distance_bins).double()

    def same(key: str) -> torch.Tensor:
        return (static[key].unsqueeze(-1) == static[key].unsqueeze(-2)).double().unsqueeze(-1)

    def one_hot(key: str, classes: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(static[key].long().clamp(max=classes - 1), classes).double()

    def scalar(key: str) -> torch.Tensor:
        return static[key].double().unsqueeze(-1)

    features = [bins(coordinates)] + ([bins(self_coordinates) * active] if legacy else [])
    features += [scalar("bond_mask"), one_hot("bond_order", config.num_bond_orders), scalar("is_aromatic")]
    features += [same("condition_index-group_1_token"), same("molecule_iid")]
    features += [] if legacy else [same("molecule_entity_id")]
    features += [one_hot("token_bond_path_length", config.token_bond_path_bins), scalar("canonical_polymer_ordering")]
    features += [] if legacy else [scalar("na_ss_pair"), scalar("na_ss_pair_mask")]
    projected = torch.cat(features, dim=-1) @ builder.pair_projection.weight.double().T
    norm = builder.pair_norm
    if isinstance(norm, torch.nn.LayerNorm):
        weight, bias = norm.weight.double(), norm.bias.double()
        normed = torch.nn.functional.layer_norm(projected, norm.normalized_shape, weight, bias, norm.eps)
    else:
        normed = torch.nn.functional.rms_norm(projected, norm.normalized_shape, norm.weight.double(), norm.eps)
    pair = normed.to(builder.pair_adaln_dtype).double()
    condition = builder.pair_condition_norm(builder.pair_condition_projection(time.to(builder.dtype)))
    scale, shift = builder.pair_adaln.modulation(condition.to(builder.pair_adaln_dtype))
    pair = torch.nn.functional.rms_norm(pair, (pair.shape[-1],), eps=builder.pair_adaln.eps)
    pair = torch.sigmoid(scale.double())[:, None, None] * pair + shift.double()[:, None, None]
    return pair * (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)


@pytest.mark.parametrize("norm", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("schema", ["sequence_probability", "legacy_self_conditioned"])
def test_pair_track_matches_the_one_hot_projection(schema: str, norm: str) -> None:
    """One kernel builds the pair track from class indices and scalars; it matches the projection of the one-hot
    features in FP64 to the AdaLN dtype's rounding, distances on a bin edge and every static channel included."""
    model, static, _state, times, _self_conditioning, mask = _unified_builder_inputs(
        input_schema=schema, factory_norm=norm
    )
    builder = model.representation_builder
    config = builder.config
    batch, tokens = mask.shape
    generator = torch.Generator(device="cuda").manual_seed(3)
    classes = {"bond_order": config.num_bond_orders, "token_bond_path_length": config.token_bond_path_bins + 3}
    for key, high in classes.items():
        static[key] = torch.randint(0, high, (batch, tokens, tokens), device="cuda", generator=generator)
    for key in ("bond_mask", "is_aromatic", "canonical_polymer_ordering", "na_ss_pair", "na_ss_pair_mask"):
        static[key] = torch.randint(0, 2, (batch, tokens, tokens), device="cuda", generator=generator).bool()
    static["na_ss_pair"] = static["na_ss_pair"].float() * 0.5
    static["molecule_iid"][:, tokens // 2 :] = 1
    static["molecule_entity_id"] = torch.zeros_like(static["molecule_iid"])
    edges = torch.linspace(config.pair_distance_min, config.pair_distance_max, config.pair_distance_bins - 1)
    coordinates = torch.randn(batch, tokens, 3, device="cuda", generator=generator)
    # Token k + 1 sits exactly one bin edge from token 0, so their distance is that edge.
    coordinates[:, 0] = 0
    coordinates[:, 1:tokens, 0] = edges[: tokens - 1].cuda()
    coordinates[:, 1:tokens, 1:] = 0
    # Token 1 sits on the first edge off the axes, where a fused multiply-add in the squared distance would
    # round past the edge into the next bin.
    coordinates[:, 1] = torch.tensor([-0.004625731613487005, -0.0016711255302652717, -0.09987898170948029])
    self_coordinates = torch.randn(batch, tokens, 3, device="cuda", generator=generator)
    with torch.inference_mode():
        parts = builder.prepare(static, mask)
        time = builder._time(times)
        for active in (False, True):
            pair = builder._pair(parts, {"bb_ca": coordinates}, time, {"bb_ca": self_coordinates}, active)
            expected = _one_hot_pair(builder, static, mask, coordinates, self_coordinates, float(active), time)
            assert pair.dtype == builder.pair_adaln_dtype
            torch.testing.assert_close(pair.double(), expected, rtol=2**-6, atol=2**-6)
            assert (pair.double() - expected).abs().mean() < 2e-3


def test_pair_track_writes_both_padded_buffers_as_the_trunk_would_pad_it() -> None:
    """Into two padded buffers, the pair track lands as the trunk's masked padding of it, bit for bit, whatever the
    buffers held."""
    from bionemo_ir._torch.layers.transformers import local_latents_transformer

    model, static, state, times, self_conditioning, mask = _unified_builder_inputs()
    builder = model.representation_builder
    batch, tokens = mask.shape
    padded = tokens + 8
    with torch.inference_mode():
        parts = builder.prepare(static, mask)
        time = builder._time(times)
        plain = builder._pair(parts, state, time, self_conditioning, False)
        buffers = tuple(
            torch.full((batch, padded, padded, plain.shape[-1]), torch.nan, device="cuda", dtype=plain.dtype)
            for _ in range(2)
        )
        written = builder._pair(parts, state, time, self_conditioning, False, out=buffers)
        expected = local_latents_transformer._masked_pair(plain, mask, padded)
    assert written.data_ptr() == buffers[0].data_ptr() and written.shape == plain.shape
    for buffer in buffers:
        assert torch.equal(buffer, expected)


def test_pair_track_reads_strided_feature_views() -> None:
    """The pair kernel reads non-dense views of its features and projection rows as it reads their dense copies."""
    model, static, state, times, self_conditioning, mask = _unified_builder_inputs()
    builder = model.representation_builder

    def strided(t: torch.Tensor) -> torch.Tensor:
        # The same values, stored with the first two axes swapped.
        return t.transpose(0, 1).contiguous().transpose(0, 1)

    with torch.inference_mode():
        parts = builder.prepare(static, mask)
        time = builder._time(times)
        features = parts.pair_features
        views = features._replace(
            classes=strided(features.classes),
            scalars=strided(features.scalars),
            weights=tuple(strided(weight) for weight in features.weights),
        )
        assert not views.classes.is_contiguous() and not views.scalars.is_contiguous()
        dense = builder._pair(parts, state, time, self_conditioning, False)
        viewed = builder._pair(dataclasses.replace(parts, pair_features=views), state, time, self_conditioning, False)
    assert torch.equal(viewed, dense)


def test_a_rollout_prepares_its_static_parts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every step of a native rollout, two trunk passes each, reuses the parts its first step built."""
    sampling = _deterministic_simulation_config()
    model, static, _state, _times, _self_conditioning, mask = _unified_builder_inputs(sampling)
    assert model.config.simulation == sampling
    init_module_weights(model)
    builder = model.representation_builder
    prepared, built = [], []
    prepare = builder.prepare
    monkeypatch.setattr(builder, "prepare", lambda *args, **kwargs: prepared.append(1) or prepare(*args, **kwargs))
    handle = builder.register_forward_pre_hook(lambda _module, _args: built.append(1))
    try:
        with torch.inference_mode():
            _run_model(model, mask, static_features=static, seed=0)
    finally:
        handle.remove()
    assert len(prepared) == 1
    assert len(built) > 1


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
        adaln_conditions.append(args[0].detach().clone())

    handles = [
        model.representation_builder.pair_condition_projection.register_forward_pre_hook(capture_projection_input),
        model.representation_builder.pair_adaln.norm_cond.register_forward_pre_hook(capture_adaln_condition),
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
    model = _small_model().cuda()
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
    model = _small_model()
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


def test_vf_tsr_scaled_score_drift_matches_the_rfd4_update() -> None:
    from bionemo_ir._torch.modules.rfproteina.vf_tsr import compute_tsr_ratio, score_to_velocity, velocity_to_score
    from bionemo_ir._torch.sampling.random import randn
    from bionemo_ir.models.rfproteina.config import RFProteinaVFStepConfig

    config = RFProteinaVFStepConfig(
        sampling_mode="vf_tsr", sc_scale_noise=0.7, tsr_k=16.0, tsr_sigma=3.0, tsr_scaled_score_drift=True
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
    std = torch.sqrt(2.0 * noise_rate * config.sc_scale_noise * delta_time)
    expected = (
        state + (score_to_velocity(state, scaled_score, time) + noise_rate * scaled_score) * delta_time + std * noise
    )
    torch.testing.assert_close(actual, expected)


def test_vf_tsr_scaled_score_drift_excludes_matched_noise() -> None:
    from bionemo_ir.models.rfproteina.config import RFProteinaVFStepConfig

    with pytest.raises(ValidationError, match="mutually exclusive"):
        RFProteinaVFStepConfig(tsr_scaled_score_drift=True, tsr_matched_noise=True)


@pytest.mark.parametrize("modality", ["bb_ca", "local_latents"])
@pytest.mark.parametrize("field", [None, "step", "reference"])
def test_simulation_rejects_unknown_nested_settings(modality: str, field: str | None) -> None:
    config = RFProteinaSimulationConfig().model_dump()
    target = config[modality] if field is None else config[modality][field]
    target["typo"] = 1
    with pytest.raises(ValueError, match="typo"):
        RFProteinaSimulationConfig.model_validate(config)
