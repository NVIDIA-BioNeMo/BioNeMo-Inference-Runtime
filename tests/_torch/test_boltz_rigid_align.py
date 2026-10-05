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
"""Numerical guards for the sync-free Boltz rigid alignment (fused kernel + torch path)."""

import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import bionemo_ir._torch.modules.boltz.loss.diffusion as boltz_diffusion
from bionemo_ir._torch.modules.boltz.loss.diffusion import horn_rotation, weighted_rigid_align
from bionemo_ir.dsl_kernels.triton.rigid_align import rigid_align_transform, supports_rigid_align

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def _inputs(
    device: str, dtype: torch.dtype = torch.float32, batch_size: int = 1, multiplicity: int = 5, num_points: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(42)
    shape = (batch_size, multiplicity, num_points)
    true_coords = torch.randn((*shape, 3), device=device, generator=generator).to(dtype)
    rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=dtype)
    translation = torch.tensor([2, -3, 1], device=device, dtype=dtype)
    pred_coords = true_coords @ rotation.T + translation
    weights = (torch.rand(shape, device=device, generator=generator) + 0.2).to(dtype)
    mask = torch.ones(shape, device=device, dtype=dtype)
    mask[..., ::7] = 0
    return true_coords, pred_coords, weights, mask


def _svd_reference(
    true_coords: torch.Tensor, pred_coords: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Kabsch through ``torch.linalg.svd`` in float64 (the previous implementation)."""
    true_coords, pred_coords, weights, mask = (t.double() for t in (true_coords, pred_coords, weights, mask))
    weights = (mask * weights).unsqueeze(-1)
    mass = weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
    true_centroid = (true_coords * weights).sum(dim=2, keepdim=True) / mass
    pred_centroid = (pred_coords * weights).sum(dim=2, keepdim=True) / mass
    true_centered = true_coords - true_centroid
    pred_centered = pred_coords - pred_centroid
    eye = torch.eye(3, dtype=torch.float64, device=true_coords.device)
    cov = (weights * pred_centered).mT @ true_centered / mass + eye * 1e-6
    u, _, vh = torch.linalg.svd(cov)
    v = vh.mH
    correction = eye.expand_as(cov).clone()
    correction[..., -1, -1] = torch.det(u @ v.mT)
    rotation = u @ correction @ v.mT
    return true_centered @ rotation.mT + pred_centroid


def _geometry(device: str, geometry: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    elif geometry == "elongated":
        true_coords *= torch.tensor([30, 0.3, 0.2], device=device)
        pred_coords = true_coords.roll(1, dims=-1) + 2
    elif geometry == "noisy_offset":
        generator = torch.Generator(device=device).manual_seed(7)
        pred_coords = pred_coords + 300 + 160 * torch.randn(pred_coords.shape, device=device, generator=generator)
    return true_coords, pred_coords, weights, mask


_GEOMETRIES = [
    "full_rank",
    "ill_conditioned",
    "planar",
    "collinear",
    "collapsed",
    "empty_mask",
    "reflection",
    "elongated",
    "noisy_offset",
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("batch_size", "multiplicity"), [(1, 5), (2, 3)])
def test_recovers_weighted_rigid_transform(device: str, dtype: torch.dtype, batch_size: int, multiplicity: int) -> None:
    inputs = _inputs(device, dtype, batch_size, multiplicity)
    aligned = weighted_rigid_align(*inputs)
    assert aligned.shape == inputs[0].shape
    assert aligned.dtype == dtype
    torch.testing.assert_close(aligned, inputs[1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_matches_svd_reference(device: str, geometry: str) -> None:
    inputs = _geometry(device, geometry)
    actual = weighted_rigid_align(*inputs)
    expected = _svd_reference(*inputs).to(actual.dtype)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_torch_path_solves_float64_in_float64(device: str) -> None:
    inputs = _inputs(device, torch.float64)
    actual = weighted_rigid_align(*inputs)
    torch.testing.assert_close(actual, _svd_reference(*inputs), atol=1e-12, rtol=1e-12)


@_CUDA
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_kernel_matches_torch_path(geometry: str) -> None:
    inputs = _geometry("cuda", geometry)
    assert supports_rigid_align(*inputs)
    fused = weighted_rigid_align(*inputs)
    with patch.object(boltz_diffusion, "supports_rigid_align", return_value=False):
        eager = weighted_rigid_align(*inputs)
    torch.testing.assert_close(fused, eager, atol=2e-5, rtol=2e-5)


def _check_large_cloud() -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    true_coords = 4000 * torch.randn((1, 1, 4096, 3), device="cuda", generator=generator)
    x, y, z = true_coords.unbind(dim=-1)
    pred_coords = torch.stack((0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y, z), dim=-1)
    pred_coords += torch.tensor([2, -3, 1], device="cuda")
    mask = torch.ones(true_coords.shape[:-1], device="cuda")
    actual = weighted_rigid_align(true_coords, pred_coords, mask, mask)
    torch.testing.assert_close(actual, pred_coords, atol=0.004, rtol=1e-6)


@_CUDA
@pytest.mark.parametrize("tf32", ["0", "1"])
def test_large_cloud_precision(tf32: str) -> None:
    # Conftest disables TF32 before CUDA initialization.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests._torch.test_boltz_rigid_align import _check_large_cloud; _check_large_cloud()",
        ],
        env={**os.environ, "NVIDIA_TF32_OVERRIDE": tf32, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": tf32},
        check=True,
    )


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


@pytest.mark.parametrize("scale", [0.0, 1e-6, 1.0, 1e3])
def test_horn_rotation_is_proper(device: str, scale: float) -> None:
    generator = torch.Generator(device=device).manual_seed(3)
    cov = torch.randn(4, 7, 3, 3, device=device, generator=generator) * scale
    rotation = horn_rotation(cov)
    assert rotation.dtype == cov.dtype
    identity = torch.eye(3, device=device).expand_as(rotation)
    torch.testing.assert_close(rotation @ rotation.mT, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.det(rotation), torch.ones(4, 7, device=device), atol=1e-6, rtol=1e-6)


@_CUDA
def test_kernel_transform_identity_on_degenerate_clouds() -> None:
    true_coords, pred_coords, weights, mask = _geometry("cuda", "empty_mask")
    rotation, shift = rigid_align_transform(true_coords, pred_coords, weights, mask)
    torch.testing.assert_close(rotation, torch.eye(3, device="cuda").expand_as(rotation), atol=0, rtol=0)
    torch.testing.assert_close(shift, torch.zeros_like(shift), atol=0, rtol=0)


def _blocking_runtime_calls(fn: Callable[[], object]) -> Counter[str]:
    """CUDA runtime calls that wait on or copy from the device, as seen by ``torch.profiler``."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    names = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "cudaMemcpyAsync")
    return Counter(event.name for event in prof.events() if event.name in names or event.name.startswith("Memcpy"))


@_CUDA
def test_kernel_has_no_host_sync() -> None:
    inputs = _inputs("cuda", num_points=4000)
    weighted_rigid_align(*inputs)
    # The profiler itself issues a device synchronize; only calls beyond that baseline count.
    baseline = _blocking_runtime_calls(lambda: None)
    actual = _blocking_runtime_calls(lambda: [weighted_rigid_align(*inputs) for _ in range(5)])
    assert actual == baseline


@_CUDA
def test_kernel_replays_under_cuda_graph() -> None:
    inputs = _inputs("cuda", num_points=300)
    static = [tensor.clone() for tensor in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            weighted_rigid_align(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        aligned = weighted_rigid_align(*static)
    shifted = inputs[0] + 1.0
    static[0].copy_(shifted)
    graph.replay()
    torch.cuda.synchronize()
    expected = _svd_reference(shifted, *inputs[1:]).float()
    torch.testing.assert_close(aligned, expected, atol=2e-4, rtol=2e-4)


@_CUDA
def test_kernel_launches_from_threads() -> None:
    batches = [_inputs("cuda", batch_size=2, multiplicity=3, num_points=200 + 64 * i) for i in range(4)]
    expected = [_svd_reference(*batch).float() for batch in batches]
    weighted_rigid_align(*batches[0])

    def run(batch: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        # The copy kernels bind the CUDA context to this thread, as a caller's own ops would.
        batch = tuple(tensor.clone() for tensor in batch)
        return [weighted_rigid_align(*batch) for _ in range(50)]

    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        results = list(pool.map(run, batches))
    torch.cuda.synchronize()
    for aligned, reference in zip(results, expected, strict=True):
        for actual in aligned:
            torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)


@_CUDA
def test_kernel_rejects_unsupported_inputs() -> None:
    true_coords, pred_coords, weights, mask = _inputs("cuda")
    assert supports_rigid_align(true_coords, pred_coords, weights, mask)
    assert not supports_rigid_align(true_coords.double(), pred_coords, weights, mask)
    assert not supports_rigid_align(true_coords.mT.contiguous().mT, pred_coords, weights, mask)
    assert not supports_rigid_align(true_coords, pred_coords, weights[..., :1], mask)
