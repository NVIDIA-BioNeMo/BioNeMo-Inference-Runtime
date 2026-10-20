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

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from bionemo_ir._torch.sampling.flow_matching import FlowMatchingSchedule

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina.config import (
        RFProteinaModalitySamplingConfig,
        RFProteinaPartialSimulationConfig,
        RFProteinaSamplingConfig,
    )


def compute_tsr_ratio(t: torch.Tensor, k: float, sigma: float) -> torch.Tensor:
    """Compute the temporal score-rescaling ratio."""
    snr = t.square() / (1.0 - t).square()
    return (snr * sigma**2 + 1.0) / (snr * sigma**2 / k + 1.0)


def get_tsr_safe_schedule(
    mode: str,
    num_steps: int,
    power: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Time schedule - Uniform, Power, or Log. Create the released inference grid without evaluating TSR at zero."""
    if mode == "uniform":
        return torch.linspace(eps, 1.0, num_steps + 1, device=device, dtype=dtype)
    if mode == "power":
        times = torch.linspace(eps ** (1.0 / power), 1.0, num_steps + 1, device=device, dtype=dtype)
        return times**power
    if mode == "log":
        if power <= 0:
            raise ValueError("log schedule power must be positive")
        times = 1.0 - torch.logspace(-power, eps, num_steps + 1, device=device, dtype=dtype).flip(0)
        times = times - (times.min() - eps)
        return times / times.max()
    raise ValueError(f"Unsupported RFProteína schedule mode: {mode}")


def get_noise_rate(
    times: torch.Tensor,
    mode: str,
    power: float,
    clamp: float | None,
    eps: float = 1e-2,
) -> torch.Tensor:
    """g(t), noise rate. Evaluate the RFProteína SDE forcing schedule."""
    if mode == "ode":
        return torch.zeros_like(times)

    safe_times = times.clamp(0.0, 1.0 - 1e-5)
    if mode == "1/t":
        rate = 1.0 / (safe_times + eps)
    elif mode == "tan":
        angle = (1.0 - safe_times) * torch.pi / 2.0
        rate = (torch.pi / 2.0) * torch.sin(angle) / (torch.cos(angle) + eps)
    elif mode == "1-t/t":
        rate = (1.0 - safe_times) / (safe_times + eps)
    else:
        raise ValueError(f"Unsupported RFProteína noise schedule: {mode}")

    if power != 1.0:
        log_rate = torch.log(rate)
        mean = log_rate.mean()
        normalized = torch.sigmoid(log_rate - mean) ** power
        rate = torch.exp(torch.logit(normalized, eps=1e-6) + mean)
    return rate.clamp(min=0.0, max=clamp)


def densify_schedule_tail(
    times: Callable[[int], torch.Tensor],
    num_steps: int,
    tail_steps: int,
    multiplier: int,
) -> torch.Tensor:
    """Sample the final ``tail_steps`` intervals at ``multiplier`` times the density.

    The returned grid still has ``num_steps`` intervals: the extra tail points
    are paid for by coarsening the head, so a schedule keeps its step budget
    while resolving the near-clean end more finely.
    """
    if tail_steps < 1:
        raise ValueError("tail_steps must be positive")
    if multiplier < 2:
        raise ValueError("tail_step_multiplier must be an integer >= 2")
    base_steps = num_steps - (multiplier - 1) * tail_steps
    if base_steps < tail_steps:
        raise ValueError(
            f"num_steps must be at least tail_steps * tail_step_multiplier ({tail_steps * multiplier}), got {num_steps}"
        )
    fine_steps = base_steps * multiplier
    tail_start = (base_steps - tail_steps) * multiplier
    indices = [*range(0, tail_start, multiplier), *range(tail_start, fine_steps + 1)]
    fine = times(fine_steps)
    return fine[torch.tensor(indices, device=fine.device)]


def create_modality_schedule(
    config: RFProteinaModalitySamplingConfig,
    num_steps: int,
    device: torch.device,
) -> FlowMatchingSchedule:
    """Build one modality's default inference schedule."""

    def grid(steps: int) -> torch.Tensor:
        return get_tsr_safe_schedule(
            config.schedule.mode,
            steps,
            config.schedule.power,
            device,
        )

    if config.schedule.tail_steps:
        times = densify_schedule_tail(
            grid,
            num_steps,
            config.schedule.tail_steps,
            config.schedule.tail_step_multiplier,
        )
    else:
        times = grid(num_steps)
    noise_rate = get_noise_rate(
        times[:-1],
        config.noise_schedule.mode,
        config.noise_schedule.power,
        config.noise_schedule.clamp,
    )
    if config.step.sampling_mode == "vf_tsr":
        ratio = compute_tsr_ratio(
            times[:-1],
            config.step.tsr_k,
            config.step.tsr_sigma,
        )
        # The TSR-scaled score drift multiplies the score by the ratio again, so the schedule is divided by
        # the ratio twice to keep the Langevin drift and noise at the temperature the update targets.
        noise_rate = noise_rate / (ratio.square() if config.step.tsr_scaled_score_drift else ratio)
    return FlowMatchingSchedule(times=times, noise_rate=noise_rate)


def _validate_partial_start_times(start_times: dict[str, float]) -> dict[str, float]:
    """Reject start times at or above 1, where the score conversion is undefined.

    Upstream ``resolve_partial_t`` used to silently clamp ``t >= 1`` on the
    schedule builder. The noising step never saw that clamp, so the integrator
    was told the state held more noise than it did. The only remaining guard is
    ``t < 1``, and it must apply to the single value both corruption and the
    schedule builder read.
    """
    at_or_above_one = {modality: time for modality, time in start_times.items() if time >= 1.0}
    if at_or_above_one:
        raise ValueError(
            "Partial-diffusion start time must be < 1 (t=1 is the clean state, and the score "
            f"conversion divides by 1 - t); resolved {at_or_above_one}. Lower partial_t_backbone / "
            "partial_t_latent, or raise partial_trajectory_fraction_steps."
        )
    return start_times


def resolve_partial_times(
    config: RFProteinaPartialSimulationConfig,
    sampling_config: RFProteinaSamplingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Resolve independent, coupled, trajectory-fraction, or inverse-folding start times.

    Coupling follows the upstream implementation: locate the first reference
    schedule index at or above the supplied time and read every modality at
    that same index. A trajectory fraction restarts on the configured schedule
    itself, at the index that leaves ``round(fraction * num_steps)`` steps, so
    the tail grid keeps the place it has in the full run. Inverse folding is
    full-noise on both modalities with the backbone resampled from the clean
    target during integration.
    """
    if config.inverse_folding:
        return _validate_partial_start_times({"bb_ca": 0.0, "local_latents": 0.0})

    backbone = config.partial_t_backbone
    latent = config.partial_t_latent
    fraction = config.partial_trajectory_fraction_steps
    if fraction is None and backbone is None and latent is None:
        return {}

    if fraction is not None:
        full = _configured_schedules(sampling_config, device)
        index = sampling_config.num_steps - partial_num_steps(config, sampling_config.num_steps)
        return _validate_partial_start_times(
            {modality: float(schedule.times[index].item()) for modality, schedule in full.items()}
        )

    if backbone is not None and latent is not None:
        return _validate_partial_start_times({"bb_ca": float(backbone), "local_latents": float(latent)})

    reference = {
        "bb_ca": create_modality_schedule(sampling_config.backbone, config.reference_num_steps, device),
        "local_latents": create_modality_schedule(sampling_config.local_latents, config.reference_num_steps, device),
    }

    if backbone is not None:
        reference_modality, reference_value = "bb_ca", backbone
    else:
        reference_modality, reference_value = "local_latents", latent
    schedule = reference[reference_modality].times
    above = schedule >= float(reference_value) - 1e-6
    index = int(above.nonzero(as_tuple=True)[0][0]) if torch.any(above) else config.reference_num_steps
    return _validate_partial_start_times(
        {modality: float(modality_schedule.times[index].item()) for modality, modality_schedule in reference.items()}
    )


def partial_num_steps(config: RFProteinaPartialSimulationConfig, configured_num_steps: int) -> int:
    """Return the executed step count for this partial-time interface."""
    fraction = config.partial_trajectory_fraction_steps
    num_steps = configured_num_steps if fraction is None else round(fraction * configured_num_steps)
    if num_steps <= 0:
        raise ValueError(
            "partial trajectory resolves to zero sampling steps; increase "
            "partial_trajectory_fraction_steps or the configured num_steps"
        )
    return num_steps


def _configured_schedules(
    sampling_config: RFProteinaSamplingConfig, device: torch.device
) -> dict[str, FlowMatchingSchedule]:
    """Both modalities' full schedules at the configured step count."""
    return {
        "bb_ca": create_modality_schedule(sampling_config.backbone, sampling_config.num_steps, device),
        "local_latents": create_modality_schedule(sampling_config.local_latents, sampling_config.num_steps, device),
    }


def create_fraction_partial_schedules(
    sampling_config: RFProteinaSamplingConfig,
    num_steps: int,
    device: torch.device,
) -> dict[str, FlowMatchingSchedule]:
    """Slice the last ``num_steps`` intervals off the configured full schedules.

    This is what a trajectory fraction integrates upstream: the tail of the full
    run, with its densified final intervals and noise rates untouched, so the
    first time is exactly the one :func:`resolve_partial_times` corrupted to.
    """
    if not 1 <= num_steps <= sampling_config.num_steps:
        raise ValueError(f"num_steps must be in [1, {sampling_config.num_steps}], got {num_steps}")
    return {
        modality: FlowMatchingSchedule(
            times=schedule.times[-(num_steps + 1) :],
            noise_rate=None if schedule.noise_rate is None else schedule.noise_rate[-num_steps:],
        )
        for modality, schedule in _configured_schedules(sampling_config, device).items()
    }


def create_partial_modality_schedule(
    config: RFProteinaModalitySamplingConfig,
    start_time: float,
    num_steps: int,
    device: torch.device,
) -> FlowMatchingSchedule:
    """Build the upstream dense-reference schedule tail ``[start_time, 1]``.

    The start time must be the same value corruption used. Do not clamp it to
    ``1 - 1/num_steps``: that made small ``partial_trajectory_fraction_steps``
    integrate a nearly-clean state as if it were much noisier.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    start_time = float(start_time)
    if start_time >= 1.0:
        raise ValueError(
            "Partial-diffusion start time must be < 1 (t=1 is the clean state, and the score "
            f"conversion divides by 1 - t); got {start_time}."
        )
    dense_steps = max(num_steps * 100, 1000)
    dense = create_modality_schedule(config, dense_steps, device)
    above = dense.times >= start_time - 1e-6
    start_index = int(above.nonzero(as_tuple=True)[0][0]) if torch.any(above) else dense_steps
    points_above = max(dense_steps - start_index, 1)
    expanded_steps = round(num_steps * dense_steps / points_above)
    full = create_modality_schedule(config, expanded_steps, device)
    return FlowMatchingSchedule(
        times=full.times[-(num_steps + 1) :],
        noise_rate=(None if full.noise_rate is None else full.noise_rate[-num_steps:]),
    )


def create_partial_schedules(
    sampling_config: RFProteinaSamplingConfig,
    start_times: dict[str, float],
    num_steps: int,
    device: torch.device,
) -> dict[str, FlowMatchingSchedule]:
    """Build equal-length partial schedule tails for both product modalities."""
    missing = {"bb_ca", "local_latents"} - set(start_times)
    if missing:
        raise ValueError(f"partial start times are missing modalities {sorted(missing)}")
    return {
        "bb_ca": create_partial_modality_schedule(sampling_config.backbone, start_times["bb_ca"], num_steps, device),
        "local_latents": create_partial_modality_schedule(
            sampling_config.local_latents, start_times["local_latents"], num_steps, device
        ),
    }
