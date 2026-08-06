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
"""Tests for Boltz ``weighted_rigid_align`` numerical robustness."""

import torch

from tensorrt_bionemo._torch.modules.boltz.loss.diffusion import weighted_rigid_align


def _random_rigid(batch=2, multiplicity=1, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(batch, multiplicity, n, 3, generator=g)
    # Build a proper rotation via QR of a random 3x3.
    a = torch.randn(batch, multiplicity, 3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    # Ensure det=+1. det is [B, M]; two unsqueezes give [B, M, 1, 1] to
    # broadcast over the 3x3. An extra unsqueeze would make rank 5 and
    # expand q to the wrong shape before the einsum.
    det = torch.det(q)
    q = q * det.sign().unsqueeze(-1).unsqueeze(-1)
    t = torch.randn(batch, multiplicity, 1, 3, generator=g)
    rotated = torch.einsum("b m n i, b m j i -> b m n j", coords, q) + t
    mask = torch.ones(batch, multiplicity, n)
    weights = torch.ones(batch, multiplicity, n)
    return coords, rotated, weights, mask, q, t


def test_weighted_rigid_align_recovers_known_transform():
    true, pred, weights, mask, _, _ = _random_rigid()
    aligned = weighted_rigid_align(true, pred, weights, mask)
    # After alignment, true should match pred up to numerical noise.
    assert torch.allclose(aligned, pred, atol=1e-4, rtol=1e-4)


def test_weighted_rigid_align_collinear_points_no_crash():
    """Collinear clouds make the 3x3 covariance rank-deficient."""
    b, m, n = 2, 1, 128
    t = torch.linspace(-10.0, 10.0, n)
    true = (
        torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)
        .view(1, 1, n, 3)
        .expand(b, m, n, 3)
        .contiguous()
    )
    pred = true + torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 1, 3)
    mask = torch.ones(b, m, n)
    weights = torch.ones(b, m, n)
    aligned = weighted_rigid_align(true, pred, weights, mask)
    assert aligned.shape == true.shape
    assert torch.isfinite(aligned).all()


def test_weighted_rigid_align_empty_mask_no_crash():
    """All-zero mask previously divided by zero → NaN → SVD LinAlgError."""
    b, m, n = 1, 1, 32
    true = torch.randn(b, m, n, 3)
    pred = torch.randn(b, m, n, 3)
    mask = torch.zeros(b, m, n)
    weights = torch.ones(b, m, n)
    aligned = weighted_rigid_align(true, pred, weights, mask)
    assert aligned.shape == true.shape
    assert torch.isfinite(aligned).all()


def test_weighted_rigid_align_large_n_scaled_coords_no_crash():
    """Large N + large coordinate magnitudes (big-complex mid-denoising)."""
    b, m, n = 1, 1, 8192
    g = torch.Generator().manual_seed(7)
    true = torch.randn(b, m, n, 3, generator=g) * 500.0
    # Near-degenerate: shrink one axis so covariance is ill-conditioned.
    true[..., 2] *= 1e-8
    pred = true + torch.randn(b, m, n, 3, generator=g) * 0.01
    mask = torch.ones(b, m, n)
    weights = torch.ones(b, m, n)
    aligned = weighted_rigid_align(true, pred, weights, mask)
    assert aligned.shape == true.shape
    assert torch.isfinite(aligned).all()
