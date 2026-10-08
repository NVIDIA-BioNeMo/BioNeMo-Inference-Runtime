# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bionemo_ir._torch.modules.openfold3.confidence import PairformerEmbedding
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir.dsl_kernels.triton import distance_embedding
from bionemo_ir.dsl_kernels.triton.distance_embedding import project_distance_bins
from bionemo_ir.dsl_kernels.triton_cache import _DRIVER_TRITON_OK

DTYPES = [torch.float32, torch.bfloat16]


def _bins(count: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.linspace(3.25, 20.75, count, device="cuda").square()
    return lower, torch.cat([lower[1:], lower.new_tensor([1e9])])


def _reference(
    rows: torch.Tensor, coordinates: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.float32):
        distances = torch.sum((rows[..., None, :] - coordinates[..., None, :, :]) ** 2, dim=-1, keepdim=True)
        return F.linear(((distances > lower) * (distances < upper)).type(rows.dtype), weight)


def _check(rows: torch.Tensor, coordinates: torch.Tensor, channels: int = 32, bins: int = 15) -> torch.Tensor:
    lower, upper = _bins(bins)
    weight = torch.randn((channels, bins), device="cuda", dtype=rows.dtype)
    actual = project_distance_bins(rows, coordinates, lower, upper, weight)
    expected = _reference(rows, coordinates, lower, upper, weight)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return actual


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bins", [15, 39])
@pytest.mark.parametrize("leading", [(), (2,), (2, 1)])
def test_distance_projection(dtype: torch.dtype, bins: int, leading: tuple[int, ...]) -> None:
    coordinates = torch.randn((*leading, 19, 3), device="cuda", dtype=dtype) * 10
    _check(coordinates[..., 1:13:2, :], coordinates, bins=bins)


@pytest.mark.parametrize("dtype", DTYPES)
def test_distance_values(dtype: torch.dtype) -> None:
    # Rare bin-edge flips hide summation-order bugs; compare distances directly.
    coordinates = torch.randn((2, 64, 3), device="cuda", dtype=dtype) * 20
    lower, upper = _bins()
    zeros, ones = torch.zeros((1, 15), device="cuda"), torch.ones((1, 1), device="cuda")
    actual = project_distance_bins(coordinates, coordinates, lower, upper, zeros, distance_weight=ones)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        expected = torch.sum((coordinates[..., None, :] - coordinates[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distance_projection_bias() -> None:
    # OpenFold2 recycling embedder layout.
    coordinates = torch.randn((2, 19, 3), device="cuda") * 10
    lower, upper = _bins()
    weight, bias = torch.randn((32, 15), device="cuda"), torch.randn(32, device="cuda")
    actual = project_distance_bins(coordinates, coordinates, lower, upper, weight, bias=bias)
    distances = torch.sum((coordinates[..., None, :] - coordinates[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    expected = F.linear(((distances > lower) * (distances < upper)).float(), weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distance_projection_euclidean() -> None:
    # Protenix confidence layout.
    coordinates = torch.randn((2, 19, 3), device="cuda") * 20
    lower = torch.arange(3.25, 52.0, 1.25, device="cuda")
    upper = torch.cat([lower[1:], lower.new_tensor([1e6])])
    weight, raw = torch.randn((32, lower.numel()), device="cuda"), torch.randn((32, 1), device="cuda")
    actual = project_distance_bins(coordinates, coordinates, lower, upper, weight, distance_weight=raw, euclidean=True)
    distances = torch.sum((coordinates[..., None, :] - coordinates[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    distances = distances.sqrt()
    expected = F.linear(((distances > lower) * (distances < upper)).float(), weight) + F.linear(distances, raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_distance_edges(dtype: torch.dtype) -> None:
    centers = torch.linspace(3.25, 20.75, 15, device="cuda", dtype=dtype)
    below = torch.nextafter(centers, torch.full_like(centers, -torch.inf))
    above = torch.nextafter(centers, torch.full_like(centers, torch.inf))
    special = centers.new_tensor([0.0, float("inf"), float("nan")])
    coordinates = torch.zeros((48, 3), device="cuda", dtype=dtype)
    coordinates[:, 0] = torch.cat([below, centers, above, special])
    _check(torch.zeros((1, 3), device="cuda", dtype=dtype), coordinates)


@pytest.mark.parametrize("tokens", [0, 1])
def test_empty_distances(tokens: int) -> None:
    coordinates = torch.randn((tokens, 3), device="cuda")
    _check(coordinates, coordinates)


def test_distance_offset_views() -> None:
    coordinates = torch.randn((20, 3), device="cuda")[1:] * 10
    lower, upper = _bins()
    lower, upper = torch.cat([lower[:1], lower])[1:], torch.cat([upper[:1], upper])[1:]
    weight = torch.randn((33, 16), device="cuda")[:, 1:]
    actual = project_distance_bins(coordinates[:7], coordinates, lower, upper, weight)
    torch.testing.assert_close(actual, _reference(coordinates[:7], coordinates, lower, upper, weight), rtol=0, atol=0)


def test_distance_large_grid() -> None:
    # More pair blocks than CUDA's 65535 grid.y limit.
    coordinates = torch.randn((2049, 3), device="cuda") * 10
    _check(coordinates[:1100], coordinates, channels=1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_driver", [False, True])
def test_distance_cache_reuses_shapes(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, use_driver: bool) -> None:
    lower, upper = _bins()
    weight = torch.randn((32, 15), device="cuda", dtype=dtype)
    signature = (dtype, dtype, torch.float32, torch.float32, dtype, dtype, dtype, torch.float32)
    constants = {
        "K": 15,
        "C": 32,
        "BK": 16,
        "BP": distance_embedding._BLOCK_PAIRS,
        "BC": 32,
        "EUCLIDEAN": False,
        "HAS_BIAS": False,
        "HAS_RAW": False,
    }
    kernel = distance_embedding._cached_embedding(torch.cuda.current_device(), signature, tuple(constants.items()))
    if use_driver:
        if not _DRIVER_TRITON_OK:
            pytest.skip("Unsupported Triton driver ABI")
        assert kernel.driver is not None
    else:
        monkeypatch.setattr(kernel, "_driver", None)

    def reject_compile(*args: object, **kwargs: object) -> None:
        pytest.fail("Distance embedding recompiled for a new shape")

    monkeypatch.setattr(distance_embedding._DistanceEmbedding, "__init__", reject_compile)
    monkeypatch.setattr(distance_embedding._distance_embedding, "run", reject_compile)
    for batch, rows, tokens in [(1, 1, 1), (2, 6, 19), (3, 17, 33)]:
        coordinates = torch.randn((batch, tokens, 3), device="cuda", dtype=dtype) * 10
        x = torch.randn((batch, rows, 3), device="cuda", dtype=dtype) * 10
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                project_distance_bins(x, coordinates, lower, upper, weight)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                captured = project_distance_bins(x, coordinates, lower, upper, weight)
            x.mul_(0.5)
            graph.replay()
        torch.testing.assert_close(captured, _reference(x, coordinates, lower, upper, weight), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_embedding_chunks(dtype: torch.dtype) -> None:
    module = PairformerEmbedding.__new__(PairformerEmbedding)
    nn.Module.__init__(module)
    module.linear_distance = nn.Linear(15, 32, bias=False, device="cuda", dtype=dtype)
    module.linear_i = nn.Linear(8, 32, bias=False, device="cuda", dtype=dtype)
    module.linear_j = nn.Linear(8, 32, bias=False, device="cuda", dtype=dtype)
    module.squared_bins, module.upper = _bins()
    si = torch.randn((2, 17, 8), device="cuda", dtype=dtype)
    pair = torch.randn((2, 17, 17, 32), device="cuda", dtype=dtype)
    coordinates = torch.randn((2, 17, 3), device="cuda", dtype=dtype) * 10
    with torch.inference_mode():
        dense = module._embed_zij_dense(si, pair, coordinates)
        chunked = module._embed_zij_chunked(si, pair, coordinates, ChunkPolicy(chunk_size=5, min_size=1))
    torch.testing.assert_close(chunked, dense, rtol=0, atol=0)
