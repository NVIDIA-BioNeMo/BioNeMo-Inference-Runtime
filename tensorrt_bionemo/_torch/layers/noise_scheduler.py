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
"""Shared AF3 EDM noise scheduling, augmentation, and sampling primitives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.layers.random_augmentation import \
    centre_random_augmentation


def create_noise_schedule(
    num_points: int,
    sigma_data: float,
    s_max: float,
    s_min: float,
    rho: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    final: str = "keep",
) -> torch.Tensor:
    """Build an AF3 rho schedule from ``s_max`` to ``s_min``.

    Args:
        num_points: number of sigma levels before any appended zero.
        sigma_data / s_max / s_min / rho: schedule parameters.
        final: ``"keep"`` (no zero), ``"zero"`` (overwrite the last with 0), or
            ``"append_zero"`` (append a trailing 0).

    Returns:
        1-D noise schedule tensor.
    """
    t = torch.arange(num_points, device=device, dtype=dtype) / (num_points - 1)
    sigmas = sigma_data * (s_max**(1 / rho) + t *
                           (s_min**(1 / rho) - s_max**(1 / rho)))**rho
    if final == "zero":
        sigmas[..., -1] = 0
    elif final == "append_zero":
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
    elif final != "keep":
        raise ValueError(f"Unknown final mode: {final!r}")
    return sigmas


class SampleDiffusion(nn.Module):
    """AF3 Algorithm 18 EDM sampler.

    Subclasses implement :meth:`denoise` and may override :meth:`augment` or
    :meth:`sample`; per-call conditioning is passed through ``ctx``.
    """

    def __init__(self,
                 *,
                 gamma0: float = 0.8,
                 gamma_min: float = 1.0,
                 noise_scale: float = 1.003,
                 step_scale: float = 1.5) -> None:
        super().__init__()
        self.gamma0 = gamma0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale

    def denoise(self, x_noisy: torch.Tensor, sigma_hat: torch.Tensor,
                ctx: dict[str, Any]) -> torch.Tensor:
        """Return denoised coordinates for scalar ``sigma_hat``."""
        raise NotImplementedError

    def augment(self, x: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        """Per-step centring + random augmentation (AF3 Alg. 19); overridable."""
        return centre_random_augmentation(x, ctx.get("atom_mask"))

    def sample(self, noise_schedule: torch.Tensor,
               coords_shape: tuple[int, ...], device: torch.device,
               dtype: torch.dtype, **ctx: Any) -> torch.Tensor:
        """Run the AF3 Algorithm 18 predictor-corrector rollout.

        Args:
            noise_schedule: ``[N_iter]`` decreasing noise levels.
            coords_shape: sampled coordinate tensor shape (incl. sample dim).
            device / dtype: for the initial + per-step noise draws.
            ctx: per-call context threaded to :meth:`denoise` / :meth:`augment`
                (for example ``atom_mask`` and trunk embeddings).
        """
        x = noise_schedule[0] * torch.randn(
            coords_shape, device=device, dtype=dtype)
        for sigma_last, c_tau in zip(noise_schedule[:-1], noise_schedule[1:]):
            x = self.augment(x, ctx).to(dtype)

            # Predictor: add noise to reach the hat noise level.
            gamma = self.gamma0 if c_tau > self.gamma_min else 0.0
            sigma_hat = sigma_last * (gamma + 1)
            x_noisy = x + self.noise_scale * torch.sqrt(
                sigma_hat**2 - sigma_last**2) * torch.randn(
                    x.shape, device=device, dtype=dtype)

            # Corrector: one Euler step from sigma_hat to c_tau.
            x_denoised = self.denoise(x_noisy, sigma_hat, ctx)
            delta = (x_noisy - x_denoised) / sigma_hat
            x = x_noisy + self.step_scale * (c_tau - sigma_hat) * delta
        return x
