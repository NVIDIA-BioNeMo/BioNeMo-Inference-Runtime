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
"""Shape/masking correctness of ``centre_random_augmentation`` (AF3 Alg. 19).

``SampleDiffusion.forward`` builds atom positions with an extra diffusion-samples
axis ``S`` — ``xl = [B, S, N_atom, 3]`` — while ``atom_mask`` stays ``[B, N_atom]``
(no ``S``). ``centre_random_augmentation`` and its ``broadcast_atom_mask`` helper
must therefore broadcast the mask across that extra axis. These are pure shape
ops (rotations/translations aside), so the tests run on CPU without weights.

Regression: before ``broadcast_atom_mask`` a bare ``atom_mask[..., None]``
right-aligned the mask's batch dim with ``xl``'s samples dim, which crashed for
``B != S`` (and silently misaligned when ``B == S``).
"""
import pytest
import torch

from tensorrt_bionemo._torch.modules.openfold3.diffusion_module import (
    broadcast_atom_mask, centre_random_augmentation)


def test_broadcast_atom_mask_matches_legacy_when_dims_align():
    """With no extra batch dim, the helper equals the old ``atom_mask[..., None]``
    (so it is a no-regression change for the ``B=1`` production path)."""
    torch.manual_seed(0)
    # matched dims: positions [B, N, 3], mask [B, N]
    pos = torch.randn(2, 5, 3)
    mask = (torch.rand(2, 5) > 0.3).float()
    assert torch.equal(broadcast_atom_mask(pos, mask), mask[..., None])

    # B=1 with a samples dim broadcasts exactly like the legacy expression too
    pos_s = torch.randn(1, 4, 5, 3)
    mask_1 = (torch.rand(1, 5) > 0.3).float()
    assert torch.equal(pos_s * broadcast_atom_mask(pos_s, mask_1),
                       pos_s * mask_1[..., None])


def test_broadcast_atom_mask_inserts_samples_axis():
    """A ``[B, S, N, 3]`` position tensor gets a ``[B, 1, N, 1]`` mask."""
    pos = torch.randn(2, 3, 5, 3)
    mask = torch.ones(2, 5)
    assert tuple(broadcast_atom_mask(pos, mask).shape) == (2, 1, 5, 1)


@pytest.mark.parametrize("B,S,N", [(1, 3, 6), (2, 3, 6), (4, 2, 7)])
def test_centre_random_augmentation_shapes_and_masks(B, S, N):
    """Runs for ``B >= 1`` with a samples axis (incl. ``B != S``), preserves the
    ``[B, S, N, 3]`` shape, and zeroes masked-out atoms."""
    gen = torch.Generator().manual_seed(1)
    xl = torch.randn(B, S, N, 3, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(2))
    atom_mask = (torch.rand(B, N,
                            generator=torch.Generator().manual_seed(3)) > 0.3
                 ).double()

    out = centre_random_augmentation(xl, atom_mask, generator=gen)

    assert tuple(out.shape) == (B, S, N, 3)
    # masked-out atoms (mask==0) must be exactly zero in the output
    masked_out = out * (1.0 - atom_mask)[:, None, :, None]
    assert masked_out.abs().max().item() == 0.0


@pytest.mark.parametrize("B,S,N", [(2, 3, 6), (4, 2, 7)])
def test_centre_random_augmentation_centers_per_batch_sample(B, S, N):
    """With ``scale_trans=0`` the masked centroid of each ``(b, s)`` output is ~0.

    Centering subtracts the per-``(b, s)`` masked mean, and a rotation preserves a
    zero mean (``mean(R @ x) = R @ mean(x)``), so a correct per-``(b, s)`` mask
    application leaves every masked centroid at the origin. A mask that misaligned
    the batch/samples axes would not.
    """
    gen = torch.Generator().manual_seed(7)
    xl = torch.randn(B, S, N, 3, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(8))
    # ensure every (b) has at least one live atom so the centroid is defined
    atom_mask = (torch.rand(B, N,
                            generator=torch.Generator().manual_seed(9)) > 0.3
                 ).double()
    atom_mask[:, 0] = 1.0

    out = centre_random_augmentation(xl, atom_mask, scale_trans=0.0,
                                     generator=gen)

    mask = atom_mask[:, None, :, None]  # [B, 1, N, 1]
    centroid = (out * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1e-12)
    assert centroid.abs().max().item() < 1e-9
