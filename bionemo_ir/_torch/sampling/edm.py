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
"""AF3-style EDM schedules and integration primitives.

This module is based on the stochastic sampler in Algorithm 2 of Karras et al.,
"Elucidating the Design Space of Diffusion-Based Generative Models" (2022):
https://arxiv.org/pdf/2206.00364.

For the paper's choice ``sigma(t) = t`` and ``s(t) = 1``, one interval is:

1. initialize ``x_0 ~ N(0, sigma_0**2 I)`` (Algorithm 2, line 2);
2. temporarily increase the noise level to
   ``sigma_hat = sigma_last * (1 + gamma)`` and inject churn noise
   (lines 4-6);
3. evaluate the denoiser ``D(x_noisy; sigma_hat)`` and form the ODE direction
   ``d = (x_noisy - D(x_noisy; sigma_hat)) / sigma_hat`` (line 7);
4. advance from ``sigma_hat`` to ``sigma_next`` with an Euler step (line 8).

``AF3EDMIntegrator`` implements these operations directly through the shared
template-method lifecycle.

This is the first-order AF3/Boltz variant, not an exact implementation of the
paper's full Algorithm 2. The second denoiser evaluation and Heun correction
from lines 9-11 are intentionally absent. ``gamma0``/``gamma_min`` implement
AF3-style churn gating instead of the paper's
``S_churn / N`` plus ``[S_min, S_max]`` rule, and ``step_scale`` can scale the
Euler update. Protein-specific rigid augmentation and hooks are also outside
Algorithm 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from bionemo_ir._torch.layers.random_augmentation import centre_random_augmentation
from bionemo_ir._torch.sampling.contracts import (
    SamplingContext,
    SamplingIntegratorConfig,
    SamplingRolloutPlan,
    SamplingScheduleConfig,
)
from bionemo_ir._torch.sampling.denoise_integrator import DenoiseIntegratorTemplate
from bionemo_ir._torch.sampling.hooks import DenoiseHookPipeline

EDMFinalMode = Literal["keep", "zero", "append_zero"]


@dataclass(frozen=True)
class EDMScheduleConfig(SamplingScheduleConfig[torch.Tensor]):
    """Parameters for the AF3 rho-discretized noise schedule.

    The interpolation has the shape of Eq. 5 in the EDM paper. TRT-BioNeMo
    additionally applies ``sigma_data`` as an overall scale and supports
    model-specific terminal policies.
    """

    sigma_data: float
    s_max: float
    s_min: float
    rho: float
    final: EDMFinalMode = "keep"

    def build(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build a schedule with exactly ``num_steps`` EDM intervals.

        ``device`` defaults to CPU when omitted.
        """
        num_points = num_steps if self.final == "append_zero" else num_steps + 1
        return create_edm_schedule(
            num_points=num_points,
            sigma_data=self.sigma_data,
            s_max=self.s_max,
            s_min=self.s_min,
            rho=self.rho,
            device=device,
            dtype=dtype,
            final=self.final,
        )


@dataclass(frozen=True)
class EDMIntegratorConfig(SamplingIntegratorConfig):
    """Parameters for the AF3 adaptation of EDM Algorithm 2.

    ``noise_scale`` corresponds to the paper's ``S_noise``. ``gamma0`` and
    ``gamma_min`` select AF3's churn amount and activation threshold.
    ``step_scale`` is an AF3 extension that scales Algorithm 2's Euler step.
    """

    gamma0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = (self.gamma0, self.gamma_min, self.noise_scale, self.step_scale)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("EDM integrator parameters must be finite")
        if self.gamma0 < 0 or self.gamma_min < 0:
            raise ValueError("gamma0 and gamma_min must be nonnegative")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be nonnegative")
        if self.step_scale <= 0:
            raise ValueError("step_scale must be positive")


@dataclass(frozen=True)
class EDMRolloutPlan(SamplingRolloutPlan[torch.Tensor]):
    """Resolved tensors and shapes for one EDM rollout.

    ``schedule[i]`` and ``schedule[i + 1]`` correspond to the paper's ``t_i``
    and ``t_{i+1}``. ``augment_coordinates`` enables a protein-specific rigid
    augmentation before churn; it is not part of EDM Algorithm 2.
    """

    schedule: torch.Tensor
    coords_shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    atom_mask: torch.Tensor | None = None
    augment_coordinates: bool = True

    def __post_init__(self) -> None:
        schedule = self.schedule
        if schedule.ndim != 1 or schedule.numel() < 2:
            raise ValueError("schedule must be one-dimensional with at least two points")
        if not bool(torch.isfinite(schedule).all()):
            raise ValueError("schedule must contain only finite values")
        if not bool((schedule >= 0).all()):
            raise ValueError("schedule must be nonnegative")
        if not bool((schedule[:-1] >= schedule[1:]).all()):
            raise ValueError("EDM schedule must be nonincreasing")
        if schedule.device != self.device:
            raise ValueError("schedule and rollout device must match")
        if len(self.coords_shape) < 2 or self.coords_shape[-1] != 3:
            raise ValueError("coords_shape must end in [N_atom, 3]")

    @property
    def num_steps(self) -> int:
        return self.schedule.numel() - 1


@dataclass
class EDMDenoiseStep:
    """Resolved state surrounding Algorithm 2 lines 4-8 for one interval."""

    step_index: int
    state: torch.Tensor
    sigma_last: torch.Tensor
    sigma_next: torch.Tensor
    sigma_hat: torch.Tensor
    noisy_state: torch.Tensor
    denoised_state: torch.Tensor | None = None


def create_edm_schedule(
    num_points: int,
    sigma_data: float,
    s_max: float,
    s_min: float,
    rho: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    final: EDMFinalMode = "keep",
) -> torch.Tensor:
    """Build the rho-discretized noise levels used by Algorithm 2.

    This follows the interpolation in EDM Eq. 5, with ``sigma_data`` as an
    additional scale. ``final`` controls whether the terminal zero from Eq. 5
    is kept, substituted, or appended for model-specific rollout conventions.
    ``device`` defaults to CPU when omitted.
    """
    if device is None:
        device = torch.device("cpu")
    min_points = 1 if final == "append_zero" else 2
    if num_points < min_points:
        raise ValueError(f"num_points must be at least {min_points}")
    if sigma_data <= 0 or s_max < 0 or s_min < 0 or rho <= 0:
        raise ValueError("sigma_data and rho must be positive; s_max and s_min nonnegative")
    if s_max < s_min:
        raise ValueError("s_max must be greater than or equal to s_min")

    if num_points == 1:
        t = torch.zeros(1, device=device, dtype=dtype)
    else:
        t = torch.arange(num_points, device=device, dtype=dtype) / (num_points - 1)
    sigmas = sigma_data * (s_max ** (1 / rho) + t * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    if final == "zero":
        sigmas[-1] = 0
    elif final == "append_zero":
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
    elif final != "keep":
        raise ValueError(f"Unknown final mode: {final!r}")
    return sigmas


def edm_churn(
    state: torch.Tensor,
    sigma_last: torch.Tensor,
    sigma_hat: torch.Tensor,
    noise: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Apply EDM Algorithm 2 lines 4-6.

    ``noise`` is standard normal here; ``noise_scale`` supplies the paper's
    ``S_noise`` standard-deviation multiplier.
    """
    variance = sigma_hat.square() - sigma_last.square()
    return state + noise_scale * torch.sqrt(variance) * noise


def edm_euler_update(
    x_noisy: torch.Tensor,
    x_denoised: torch.Tensor,
    sigma_hat: torch.Tensor,
    sigma_next: torch.Tensor,
    step_scale: float,
) -> torch.Tensor:
    """Apply the derivative and Euler update from Algorithm 2 lines 7-8.

    ``x_denoised`` is ``D_theta(x_noisy; sigma_hat)``. A ``step_scale`` of
    one reproduces the paper's first-order update; AF3 uses a configurable
    scale. This function does not apply Algorithm 2's Heun correction.
    """
    direction = (x_noisy - x_denoised) / sigma_hat
    return x_noisy + step_scale * (sigma_next - sigma_hat) * direction


class AF3EDMIntegrator(
    DenoiseIntegratorTemplate[EDMRolloutPlan, torch.Tensor, EDMDenoiseStep, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Run the tensor AF3 first-order adaptation of EDM Algorithm 2."""

    def __init__(
        self,
        config: EDMIntegratorConfig,
        hook_pipeline: DenoiseHookPipeline[EDMDenoiseStep] | None = None,
    ) -> None:
        super().__init__(hook_pipeline)
        self.config = config

    def num_steps(self, plan: EDMRolloutPlan) -> int:
        return plan.num_steps

    def initialize(self, plan: EDMRolloutPlan, context: SamplingContext) -> torch.Tensor:
        # Algorithm 2, line 2: x_0 ~ N(0, sigma_0^2 I).
        noise = torch.randn(plan.coords_shape, device=plan.device, dtype=plan.dtype, generator=context.generator)
        return plan.schedule[0] * noise

    def prepare_denoise_step(
        self,
        step_index: int,
        state: torch.Tensor,
        plan: EDMRolloutPlan,
        context: SamplingContext,
    ) -> EDMDenoiseStep:
        if plan.augment_coordinates:
            state = centre_random_augmentation(
                state,
                plan.atom_mask,
                generator=context.generator,
            ).to(plan.dtype)

        sigma_last = plan.schedule[step_index]
        sigma_next = plan.schedule[step_index + 1]
        # Algorithm 2, line 5: sigma_hat = sigma_last * (1 + gamma).
        gamma = torch.where(
            sigma_next > self.config.gamma_min, sigma_next.new_tensor(self.config.gamma0), sigma_next.new_zeros(())
        )
        sigma_hat = sigma_last * (gamma + 1)
        # Algorithm 2, lines 4 and 6: sample and inject churn noise.
        noise = torch.randn(state.shape, device=plan.device, dtype=plan.dtype, generator=context.generator)
        x_noisy = edm_churn(state, sigma_last, sigma_hat, noise, self.config.noise_scale)
        return EDMDenoiseStep(
            step_index=step_index,
            state=state,
            sigma_last=sigma_last,
            sigma_next=sigma_next,
            sigma_hat=sigma_hat,
            noisy_state=x_noisy,
        )

    def denoiser_inputs(self, step: EDMDenoiseStep) -> tuple[torch.Tensor, torch.Tensor]:
        return step.noisy_state, step.sigma_hat

    def attach_prediction(self, step: EDMDenoiseStep, prediction: torch.Tensor) -> EDMDenoiseStep:
        step.denoised_state = prediction
        return step

    def complete_step(self, step: EDMDenoiseStep) -> torch.Tensor:
        if step.denoised_state is None:
            raise RuntimeError("EDM step is missing its denoised prediction")
        return edm_euler_update(
            step.noisy_state, step.denoised_state, step.sigma_hat, step.sigma_next, self.config.step_scale
        )
