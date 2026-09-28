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
"""Numerical guards for the default small-matrix rigid-alignment SVD."""

import pytest
import torch

from bionemo_ir._torch.modules.boltz.loss.diffusion import weighted_rigid_align


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required to compare cuSOLVER drivers")
    return request.param


def _inputs(
    device: str, dtype: torch.dtype = torch.float32, batch_size: int = 1, multiplicity: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(42)
    shape = (batch_size, multiplicity, 32)
    true_coords = torch.randn((*shape, 3), device=device, generator=generator).to(dtype)
    rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=dtype)
    translation = torch.tensor([2, -3, 1], device=device, dtype=dtype)
    pred_coords = true_coords @ rotation.T + translation
    weights = (torch.rand(shape, device=device, generator=generator) + 0.2).to(dtype)
    mask = torch.ones(shape, device=device, dtype=dtype)
    mask[..., ::7] = 0
    return true_coords, pred_coords, weights, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("batch_size", "multiplicity"), [(1, 5), (2, 3)])
def test_recovers_weighted_rigid_transform(device: str, dtype: torch.dtype, batch_size: int, multiplicity: int) -> None:
    inputs = _inputs(device, dtype, batch_size, multiplicity)
    aligned = weighted_rigid_align(*inputs)
    assert aligned.shape == inputs[0].shape
    assert aligned.dtype == dtype
    torch.testing.assert_close(aligned, inputs[1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "geometry", ["full_rank", "ill_conditioned", "planar", "collinear", "collapsed", "empty_mask", "reflection"]
)
def test_matches_gesvd_reference(device: str, geometry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    true_coords, pred_coords, weights, mask = _inputs(device)
    if geometry == "ill_conditioned":
        true_coords *= torch.tensor([1, 1e-3, 1e-6], device=device)
        pred_coords = true_coords.roll(1, dims=-1) + 2
    elif geometry == "planar":
        true_coords[..., 2] = 0
        pred_coords = true_coords.roll(1, dims=-1) + 2
    elif geometry == "collinear":
        true_coords[..., 1:] = 0
        pred_coords = true_coords.roll(1, dims=-1) + 2
    elif geometry == "collapsed":
        true_coords.fill_(1)
        pred_coords.fill_(2)
    elif geometry == "empty_mask":
        mask.zero_()
    elif geometry == "reflection":
        true_coords *= torch.tensor([1, 2, 4], device=device)
        pred_coords = true_coords * torch.tensor([-1, 1, 1], device=device) + 2
    inputs = (true_coords, pred_coords, weights, mask)
    actual = weighted_rigid_align(*inputs)
    original_svd = torch.linalg.svd

    def gesvd(matrix: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return original_svd(matrix, **(kwargs | {"driver": "gesvd" if matrix.is_cuda else None}))

    with monkeypatch.context() as context:
        context.setattr(torch.linalg, "svd", gesvd)
        expected = weighted_rigid_align(*inputs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("reflected", [False, True])
def test_rotation_is_proper_and_orthogonal(device: str, reflected: bool) -> None:
    true_coords, pred_coords, weights, mask = _inputs(device)
    if reflected:
        pred_coords[..., 0] *= -1
    aligned = weighted_rigid_align(true_coords, pred_coords, weights, mask)
    homogeneous = torch.cat([true_coords, torch.ones_like(true_coords[..., :1])], dim=-1)
    transform = torch.linalg.lstsq(homogeneous, aligned).solution[..., :3, :]
    identity = torch.eye(3, device=device).expand_as(transform)
    torch.testing.assert_close(transform @ transform.mT, identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.det(transform), torch.ones_like(weights[..., 0]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_preserves_inputs_rng_and_dtype(device: str, dtype: torch.dtype) -> None:
    inputs = _inputs(device, dtype)
    saved = tuple(value.clone() for value in inputs)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    aligned = weighted_rigid_align(*inputs)
    repeated = weighted_rigid_align(*inputs)
    assert aligned.dtype == dtype
    assert torch.isfinite(aligned).all()
    torch.testing.assert_close(aligned, repeated, atol=0, rtol=0)
    for actual, expected in zip(inputs, saved, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


def test_requests_default_driver_and_preserves_covariance(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(device)
    true_coords, pred_coords, weights, mask = inputs
    effective_weights = (weights * mask).unsqueeze(-1)
    mass = effective_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
    true_centered = true_coords - (true_coords * effective_weights).sum(dim=2, keepdim=True) / mass
    pred_centered = pred_coords - (pred_coords * effective_weights).sum(dim=2, keepdim=True) / mass
    expected = (effective_weights * pred_centered).mT @ true_centered / mass
    expected = expected.float() + torch.eye(3, device=device) * 1e-6
    original_svd = torch.linalg.svd
    calls = []

    def record(matrix: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(matrix.clone())
        assert kwargs.get("driver") is None
        return original_svd(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", record)
    weighted_rigid_align(*inputs)
    assert len(calls) == 1
    assert calls[0].shape == (1, 5, 3, 3)
    assert calls[0].dtype == torch.float32
    torch.testing.assert_close(calls[0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("failure", [torch.linalg.LinAlgError, RuntimeError])
def test_svd_failure_keeps_translation_fallback(
    device: str, failure: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    true_coords, pred_coords, weights, mask = _inputs(device)
    effective_weights = (weights * mask).unsqueeze(-1)
    mass = effective_weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
    expected = (
        true_coords
        - (true_coords * effective_weights).sum(dim=2, keepdim=True) / mass
        + (pred_coords * effective_weights).sum(dim=2, keepdim=True) / mass
    )

    def fail(matrix: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise failure("injected solver failure")

    monkeypatch.setattr(torch.linalg, "svd", fail)
    aligned = weighted_rigid_align(true_coords, pred_coords, weights, mask)
    torch.testing.assert_close(aligned, expected, atol=0, rtol=0)
