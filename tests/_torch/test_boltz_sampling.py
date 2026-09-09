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
"""Focused guards for Boltz's specialized EDM rollout."""

from math import sqrt
from types import SimpleNamespace

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.random_augmentation import random_rotations
from bionemo_ir._torch.modules.boltz import structure as boltz_structure
from bionemo_ir._torch.modules.boltz.physical.steering import BoltzSteeringParams
from bionemo_ir._torch.modules.boltz.structure import (
    BoltzDiffusionSampler,
    BoltzEDMDenoiseStep,
    BoltzEDMIntegrator,
)
from bionemo_ir._torch.sampling import (
    DenoiseHookPipeline,
    DenoiseIntegratorTemplate,
    EDMIntegratorConfig,
    SamplingContext,
)


class _RecordBeforeDenoise:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def before_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        del context
        self.events.append("before")
        return step


class _RecordAfterDenoise:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def after_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        del context
        self.events.append("after")
        return step


class _StubScoreModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.dtype = torch.float32


class _StubBoltzDiffusionSampler(BoltzDiffusionSampler):
    """Small sampler-only fixture that avoids constructing the score network."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.diffusion_module = _StubScoreModel()
        self.runner = boltz_structure.GenerativeRunner()
        self.gamma0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.0
        self.edm_integrator_config = EDMIntegratorConfig(
            gamma0=self.gamma0,
            gamma_min=self.gamma_min,
            noise_scale=self.noise_scale,
            step_scale=self.step_scale,
        )
        self.sigma_min = 4e-4
        self.sigma_max = 2.0
        self.sigma_data = 1.0
        self.rho = 7.0
        self.num_sampling_steps = 4
        self.version = "v2"
        self.alignment_reverse_diff = False
        self.coordinate_augmentation = True
        self.out_token_feat_update = None
        self.events: list[str] = []
        self.record_hooks = False

    def build_denoise_hook_pipeline(self, **kwargs) -> tuple[DenoiseHookPipeline, int]:
        pipeline, multiplicity = super().build_denoise_hook_pipeline(**kwargs)
        if not self.record_hooks:
            return pipeline, multiplicity
        return DenoiseHookPipeline(
            (*pipeline.before_denoise_hooks, _RecordBeforeDenoise(self.events)),
            (*pipeline.after_denoise_hooks, _RecordAfterDenoise(self.events)),
        ), multiplicity

    def denoise(
        self,
        *,
        noised_atom_coords: torch.Tensor,
        **_kwargs,
    ) -> tuple[torch.Tensor, None]:
        if self.record_hooks:
            self.events.append("network")
        return 0.7 * noised_atom_coords, None


def _sampler_config() -> SimpleNamespace:
    atom_diffusion = SimpleNamespace(
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.0,
        sigma_min=4e-4,
        sigma_max=2.0,
        sigma_data=1.0,
        rho=7.0,
        P_mean=-1.2,
        P_std=1.5,
        num_sampling_steps=4,
        coordinate_augmentation=True,
        version="v2",
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        accumulate_token_repr=False,
    )
    score_model = SimpleNamespace(dim_fourier=256, token_s=384)
    return SimpleNamespace(atom_diffusion=atom_diffusion, score_model=score_model)


def test_boltz_sampler_owns_injected_diffusion_module():
    diffusion_module = _StubScoreModel()
    sampler = BoltzDiffusionSampler(_sampler_config(), diffusion_module)

    assert sampler.diffusion_module is diffusion_module
    assert "score_model" not in sampler._modules


def test_boltz_sampler_maps_converted_score_model_weights(monkeypatch):
    sampler = BoltzDiffusionSampler(_sampler_config(), _StubScoreModel())
    converted_weight = object()
    captured = {}

    def record_weights(module, weights):
        assert module is sampler
        captured.update(weights)
        return set(weights)

    monkeypatch.setattr(boltz_structure, "recursive_calling_load_weights", record_weights)
    sampler.load_weights({"score_model.token_transformer": converted_weight})

    assert captured == {"diffusion_module.token_transformer": converted_weight}


def test_boltz_discovery_aliases_target_sampler_diffusion_module():
    from bionemo_ir.models.boltz1.modeling import Boltz1
    from bionemo_ir.models.boltz2.modeling import Boltz2

    for model_cls in (Boltz1, Boltz2):
        aliases = model_cls.GRAPH_OPT_ENABLED_MODULES
        assert aliases["token_transformer"] == ("diffusion_sampler.diffusion_module.token_transformer")
        assert aliases["diffusion_module"] == "diffusion_sampler.diffusion_module"


def _network_condition() -> dict[str, torch.Tensor]:
    return {
        "q": torch.zeros(1),
        "c": torch.zeros(1),
        "atom_enc_bias": torch.zeros(1),
        "token_trans_bias": torch.zeros(1),
        "atom_dec_bias": torch.zeros(1),
    }


def _reference_sample(
    sampler: _StubBoltzDiffusionSampler, seed: int, num_steps: int, multiplicity: int, num_atoms: int
) -> torch.Tensor:
    generator = torch.Generator(device=sampler.device)
    generator.manual_seed(seed)
    sigmas = sampler.sample_schedule(num_steps)
    gammas = torch.where(sigmas > sampler.gamma_min, sampler.gamma0, 0.0)
    coords_shape = (1, multiplicity, num_atoms, 3)
    atom_coords = sigmas[0] * torch.randn(coords_shape, generator=generator)
    atom_coords_denoised = None

    for sigma_last, sigma_next, gamma in zip(sigmas[:-1], sigmas[1:], gammas[1:], strict=True):
        rotation = random_rotations(
            multiplicity, device=atom_coords.device, dtype=atom_coords.dtype, generator=generator
        ).view(1, multiplicity, 3, 3)
        translation = torch.randn((1, multiplicity, 1, 3), dtype=atom_coords.dtype, generator=generator)
        atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
        atom_coords = torch.einsum("bmnd,bmds->bmns", atom_coords, rotation) + translation
        if atom_coords_denoised is not None:
            atom_coords_denoised = atom_coords_denoised - atom_coords_denoised.mean(dim=-2, keepdims=True)
            atom_coords_denoised = torch.einsum("bmnd,bmds->bmns", atom_coords_denoised, rotation) + translation

        sigma_last = sigma_last.item()
        sigma_next = sigma_next.item()
        sigma_hat = sigma_last * (1 + gamma.item())
        noise_variance = sampler.noise_scale**2 * (sigma_hat**2 - sigma_last**2)
        noise = sqrt(noise_variance) * torch.randn(coords_shape, generator=generator)
        atom_coords_noisy = atom_coords + noise
        atom_coords_denoised = 0.7 * atom_coords_noisy
        atom_coords = (
            atom_coords_noisy
            + sampler.step_scale * (sigma_next - sigma_hat) * (atom_coords_noisy - atom_coords_denoised) / sigma_hat
        )
    return atom_coords


def test_boltz_specialized_sampler_uses_explicit_seed():
    sampler = _StubBoltzDiffusionSampler()
    features = {"atom_pad_mask": torch.ones(1, 5)}
    kwargs = {
        "s_trunk": torch.zeros(1),
        "s_inputs": torch.zeros(1),
        "num_sampling_steps": 4,
        "multiplicity": 2,
        "max_parallel_samples": 1,
        "feature_dict": features,
        "sampling_seed": 17,
    }

    first = sampler(network_condition_kwargs=_network_condition(), **kwargs)
    torch.manual_seed(123456)
    second = sampler(network_condition_kwargs=_network_condition(), **kwargs)

    first_coords = first["sample_atom_coords"]
    assert first_coords.shape == (1, 2, 5, 3)
    assert torch.isfinite(first_coords).all()
    assert first["diff_token_repr"] is None
    torch.testing.assert_close(first_coords, second["sample_atom_coords"])
    torch.testing.assert_close(
        first_coords,
        _reference_sample(sampler, seed=17, num_steps=4, multiplicity=2, num_atoms=5),
    )

    graph_safe_kwargs = {**kwargs, "sampling_seed": None}
    torch.manual_seed(19)
    graph_safe_first = sampler(network_condition_kwargs=_network_condition(), **graph_safe_kwargs)
    torch.manual_seed(19)
    graph_safe_second = sampler(network_condition_kwargs=_network_condition(), **graph_safe_kwargs)
    torch.testing.assert_close(graph_safe_first["sample_atom_coords"], graph_safe_second["sample_atom_coords"])


def test_boltz_implements_template_directly_and_orders_hooks():
    sampler = _StubBoltzDiffusionSampler()
    integrator = sampler.build_edm_integrator(DenoiseHookPipeline())
    assert isinstance(integrator, BoltzEDMIntegrator)
    assert isinstance(integrator, DenoiseIntegratorTemplate)

    sampler.record_hooks = True
    sampler(
        s_trunk=torch.zeros(1),
        s_inputs=torch.zeros(1),
        num_sampling_steps=2,
        multiplicity=1,
        max_parallel_samples=1,
        network_condition_kwargs=_network_condition(),
        feature_dict={"atom_pad_mask": torch.ones(1, 3)},
        sampling_seed=5,
    )

    assert sampler.events == ["before", "network", "after"] * 2


def test_potential_guidance_hook_owns_particle_resampling(monkeypatch):
    monkeypatch.setattr(boltz_structure, "get_potentials", lambda *_args, **_kwargs: [])
    sampler = _StubBoltzDiffusionSampler()
    steering = BoltzSteeringParams(
        fk_steering=True, num_particles=2, physical_guidance_update=False, contact_guidance_update=False
    )
    output = sampler(
        s_trunk=torch.zeros(1),
        s_inputs=torch.zeros(1),
        num_sampling_steps=2,
        multiplicity=2,
        max_parallel_samples=2,
        steering_args=steering,
        network_condition_kwargs=_network_condition(),
        feature_dict={"atom_pad_mask": torch.ones(1, 3)},
        sampling_seed=11,
    )

    # The hook expands to particles internally and collapses them on the last
    # step, so the public sample multiplicity remains unchanged.
    assert output["sample_atom_coords"].shape == (1, 2, 3, 3)


def test_coordinate_augmentation_gates_the_rigid_hook():
    sampler = _StubBoltzDiffusionSampler()
    kwargs = {
        "steering_args": None,
        "atom_mask": torch.ones(1, 3),
        "multiplicity": 1,
        "num_sampling_steps": 2,
        "feature_dict": {"atom_pad_mask": torch.ones(1, 3)},
    }

    pipeline_on, _ = sampler.build_denoise_hook_pipeline(**kwargs)
    assert isinstance(pipeline_on.before_denoise_hooks[0], boltz_structure.BoltzRandomRigidAugmentationHook)
    assert isinstance(pipeline_on.before_denoise_hooks[1], boltz_structure.PotentialGuidanceHook)
    assert isinstance(pipeline_on.before_denoise_hooks[2], boltz_structure.BoltzEDMChurnHook)

    sampler.coordinate_augmentation = False
    pipeline_off, _ = sampler.build_denoise_hook_pipeline(**kwargs)
    assert not any(
        isinstance(hook, boltz_structure.BoltzRandomRigidAugmentationHook) for hook in pipeline_off.before_denoise_hooks
    )
    assert isinstance(pipeline_off.before_denoise_hooks[0], boltz_structure.PotentialGuidanceHook)
    assert isinstance(pipeline_off.before_denoise_hooks[1], boltz_structure.BoltzEDMChurnHook)
