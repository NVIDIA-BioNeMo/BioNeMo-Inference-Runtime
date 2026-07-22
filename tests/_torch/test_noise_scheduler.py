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
``_torch/layers/noise_scheduler.py`` (``create_noise_schedule`` +
``centre_random_augmentation`` + ``edm_sample``). Those models have no separate
sampler unit tests, so these tests pin the shared primitives bit-exactly to each
model's original formula / loop (fixed seed), guarding the refactor.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from tensorrt_bionemo._torch.layers.noise_scheduler import (
    SampleDiffusion, create_noise_schedule)
from tensorrt_bionemo._torch.layers.random_augmentation import (
    centre_random_augmentation, quaternion_to_matrix, random_quaternions)

_SD, _SMAX, _SMIN, _RHO = 16.0, 160.0, 4e-4, 7.0


def test_schedule_openfold3():
    """OF3 ``create_noise_schedule``: n+1 points, no final zero."""
    n = 20
    t = torch.arange(0, 1 + n, dtype=torch.float32) / n
    exp = _SD * (_SMAX**(1 / _RHO) + t *
                 (_SMIN**(1 / _RHO) - _SMAX**(1 / _RHO)))**_RHO
    got = create_noise_schedule(num_points=n + 1,
                                sigma_data=_SD,
                                s_max=_SMAX,
                                s_min=_SMIN,
                                rho=_RHO,
                                final="keep")
    torch.testing.assert_close(got, exp)
    assert got.shape == (n + 1, ) and got[-1] > 0


def test_schedule_protenix():
    """Protenix ``InferenceNoiseScheduler``: N+1 points, final -> 0."""
    n = 20
    step = torch.arange(n + 1, dtype=torch.float32) / n
    exp = _SD * (_SMAX**(1 / _RHO) + step *
                 (_SMIN**(1 / _RHO) - _SMAX**(1 / _RHO)))**_RHO
    exp[-1] = 0
    got = create_noise_schedule(num_points=n + 1,
                                sigma_data=_SD,
                                s_max=_SMAX,
                                s_min=_SMIN,
                                rho=_RHO,
                                final="zero")
    torch.testing.assert_close(got, exp)
    assert got[-1] == 0


def test_schedule_boltz():
    """Boltz ``sample_schedule``: M points then an appended 0."""
    m = 20
    steps = torch.arange(m, dtype=torch.float32)
    exp = (_SMAX**(1 / _RHO) + steps / (m - 1) *
           (_SMIN**(1 / _RHO) - _SMAX**(1 / _RHO)))**_RHO * _SD
    exp = F.pad(exp, (0, 1), value=0.0)
    got = create_noise_schedule(num_points=m,
                                sigma_data=_SD,
                                s_max=_SMAX,
                                s_min=_SMIN,
                                rho=_RHO,
                                final="append_zero")
    torch.testing.assert_close(got, exp)
    assert got.shape == (m + 1, ) and got[-1] == 0


def _oss_centre(xl, atom_mask, scale_trans=1.0):
    """Inline reproduction of the OF3 / Protenix ``centre_random_augmentation``
    (AF3 Alg. 19) with the same RNG call order (quaternion rots, then trans)."""
    n = math.prod(xl.shape[:-2])
    rots = quaternion_to_matrix(
        random_quaternions(n, dtype=xl.dtype,
                           device=xl.device)).reshape(*xl.shape[:-2], 3, 3)
    trans = scale_trans * torch.randn(
        (*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device)
    mean = (xl * atom_mask[..., None]).sum(
        -2, keepdim=True) / atom_mask[..., None].sum(
            -2, keepdim=True).clamp(min=1e-7)
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


class _StubSampler(SampleDiffusion):
    """Minimal SampleDiffusion subclass with a deterministic denoiser."""

    def denoise(self, x_noisy, sigma_hat, ctx):
        return x_noisy * 0.7 - 0.1


@pytest.mark.parametrize("final", ["zero", "keep", "append_zero"])
def test_sample_diffusion_matches_reference(final):
    """``SampleDiffusion.sample`` is bit-exact to an inline AF3 Alg. 18 loop."""
    schedule = create_noise_schedule(num_points=6,
                                     sigma_data=_SD,
                                     s_max=_SMAX,
                                     s_min=_SMIN,
                                     rho=_RHO,
                                     final=final)
    coords_shape = (2, 3, 8, 3)
    mask = torch.ones(2, 1, 8)
    g0, gmin, nscale, sscale = 0.8, 1.0, 1.003, 1.5
    sampler = _StubSampler(gamma0=g0,
                           gamma_min=gmin,
                           noise_scale=nscale,
                           step_scale=sscale)

    torch.manual_seed(123)
    got = sampler.sample(schedule,
                         coords_shape,
                         torch.device("cpu"),
                         torch.float32,
                         atom_mask=mask)

    torch.manual_seed(123)
    x = schedule[0] * torch.randn(coords_shape)
    for c_last, c_tau in zip(schedule[:-1], schedule[1:]):
        x = centre_random_augmentation(x, mask=mask)
        gamma = g0 if c_tau > gmin else 0.0
        sh = c_last * (gamma + 1)
        x_noisy = x + nscale * torch.sqrt(sh**2 - c_last**2) * torch.randn(
            x.shape)
        delta = (x_noisy - (x_noisy * 0.7 - 0.1)) / sh
        x = x_noisy + sscale * (c_tau - sh) * delta
    torch.testing.assert_close(got, x)


if __name__ == "__main__":
    test_schedule_openfold3()
    test_schedule_protenix()
    test_schedule_boltz()
    test_centre_random_augmentation_matches_oss()
    test_centre_random_augmentation_is_rigid()
    for _f in ("zero", "keep", "append_zero"):
        test_sample_diffusion_matches_reference(_f)
    print("OK")
