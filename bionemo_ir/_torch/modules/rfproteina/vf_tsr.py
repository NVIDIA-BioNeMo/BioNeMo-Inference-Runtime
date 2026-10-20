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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.sampling.flow_matching import velocity_to_clean_endpoint
from bionemo_ir._torch.sampling.random import randn

from .schedules import compute_tsr_ratio

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import RFProteinaVFStepConfig


def velocity_to_clean(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Recover the clean endpoint from RFProteína's velocity parameterization."""
    clean = velocity_to_clean_endpoint(state, velocity, time)
    return clean * mask.unsqueeze(-1).to(clean.dtype)


def _velocity_to_score(state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    return (time[..., None, None] * velocity - state) / (1.0 - time)[..., None, None]


def _score_to_velocity(state: torch.Tensor, score: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    time = time[..., None, None]
    return (state + (1.0 - time) * score) / time


def velocity_to_score(state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Convert velocity to the score of the linear interpolant."""
    if not torch.all(time < 1.0):
        raise ValueError("velocity_to_score requires time < 1")
    return _velocity_to_score(state, velocity, time)


def score_to_velocity(state: torch.Tensor, score: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Convert a linear-interpolant score back to velocity."""
    if not torch.all(time > 0.0):
        raise ValueError("score_to_velocity requires time > 0")
    return _score_to_velocity(state, score, time)


def _masked_center(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_value = mask.unsqueeze(-1).to(x.dtype)
    count = mask_value.sum(dim=-2, keepdim=True).clamp_min(1.0)
    mean = (x * mask_value).sum(dim=-2, keepdim=True) / count
    return (x - mean) * mask_value


def velocity_flow_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    delta_time: torch.Tensor,
    noise_rate: torch.Tensor,
    mask: torch.Tensor,
    config: RFProteinaVFStepConfig,
    generator: torch.Generator | None = None,
    time_value: float | None = None,
) -> torch.Tensor:
    """Advance one modality by the released Euler or VF-TSR update.

    ``time_value`` is this step's time as a host float. Every row shares one
    time, and the ODE window is decided on the host, so supplying it keeps the
    step free of device-to-host reads; omitting it reads the value back and
    costs one synchronization per call.
    """
    mode = config.sampling_mode
    if mode == "vf":
        delta = velocity * delta_time
    elif mode == "vf_tsr":
        if time_value is None:
            time_value = float(time.reshape(-1)[0])
        if not 0.0 < time_value < 1.0:
            raise ValueError(f"the VF-TSR update requires time in (0, 1), got {time_value}")
        score = _velocity_to_score(state, velocity, time)
        ratio = compute_tsr_ratio(time, config.tsr_k, config.tsr_sigma)
        scaled_velocity = _score_to_velocity(state, score * ratio[..., None, None], time)
        if time_value > config.ode_upper or time_value < config.ode_lower:
            delta = scaled_velocity * delta_time
        else:
            noise = randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)
            # Matched noise tracks the score the TSR ratio already scaled, so the
            # injected noise follows the temperature instead of a fixed setting.
            noise_scale = config.score_scale / ratio if config.tsr_matched_noise else config.noise_scale
            if config.tsr_scaled_score_drift:
                drift_score = score * ratio[..., None, None]
            else:
                drift_score = score * config.score_scale
            variance = 2.0 * noise_rate * noise_scale * delta_time
            std = torch.sqrt(variance.to(state.device, torch.float32)).to(state.dtype)
            if std.ndim:
                std = std.reshape(*std.shape, *((1,) * (state.ndim - std.ndim)))
            delta = (scaled_velocity + noise_rate * drift_score) * delta_time + std * noise
    else:
        raise ValueError(f"Unsupported RFProteína sampling mode: {mode}")

    updated = (state + delta) * mask.unsqueeze(-1).to(state.dtype)
    if config.center_every_step:
        updated = _masked_center(updated, mask)
    return updated
