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

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.graph_optimization.cudnn_graph import (
    CudnnGraphCache,
    can_use_cudnn_graph,
    cudnn_add_add_mask,
    cudnn_add_mask,
    cudnn_linear_mask,
    cudnn_linear_mask_residual,
    cudnn_linear_relu,
    cudnn_linear_residual,
    cudnn_scale_shift_mask,
    prepare_cudnn_linear_mask,
    prepare_cudnn_linear_relu,
)
from bionemo_ir._torch.graph_optimization.cudnn_graph import ops as cudnn_graph_ops


def test_cudnn_graph_cache_reuses_signatures_and_evicts_lru() -> None:
    cache = CudnnGraphCache[object]("test graph", max_size=2)
    builds = 0

    def build() -> object:
        nonlocal builds
        builds += 1
        return object()

    first = torch.empty(2, 3, device="cuda")
    second = torch.empty(3, 4, device="cuda")
    third = torch.empty(4, 5, device="cuda")

    first_plan = cache.get_or_create((first,), build)
    assert cache.get_or_create((first,), build) is first_plan
    cache.get_or_create((second,), build)
    cache.get_or_create((third,), build)
    assert cache.get_or_create((first,), build) is not first_plan
    assert builds == 4


def _released_pair_linear_operands(
    input_dim: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_1 = (
        torch.randn(hidden_dim, input_dim, device=device, dtype=dtype)
        .mul_(input_dim**-0.5)
        .unsqueeze(0)
        .transpose(-1, -2)
    )
    bias_1 = torch.randn(1, 1, hidden_dim, device=device, dtype=dtype).mul_(0.01)
    weight_2 = (
        torch.randn(input_dim, hidden_dim, device=device, dtype=dtype)
        .mul_(hidden_dim**-0.5)
        .unsqueeze(0)
        .transpose(-1, -2)
    )
    bias_2 = torch.randn(1, 1, input_dim, device=device, dtype=dtype).mul_(0.01)
    return weight_1, bias_1, weight_2, bias_2


def _clear_linear_plan_caches() -> None:
    for cache in (
        cudnn_graph_ops._DYNAMIC_LINEAR_RELU_CACHE,
        cudnn_graph_ops._DYNAMIC_LINEAR_MASK_CACHE,
        cudnn_graph_ops._LINEAR_RELU_CACHE,
        cudnn_graph_ops._LINEAR_MASK_CACHE,
        cudnn_graph_ops._LINEAR_RESIDUAL_CACHE,
        cudnn_graph_ops._LINEAR_MASK_RESIDUAL_CACHE,
    ):
        cache.clear()


def test_linear_graphs_build_one_static_plan_per_row_count() -> None:
    """The shared callables stay on static plans and never reach a dynamic one."""
    input_dim, hidden_dim = 384, 768
    _clear_linear_plan_caches()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    weight_1, bias_1, weight_2, bias_2 = _released_pair_linear_operands(input_dim, hidden_dim, device, dtype)

    row_counts = (64 * 64, 128 * 128)
    with torch.inference_mode():
        for rows in row_counts:
            value = torch.randn(1, rows, input_dim, device=device, dtype=dtype)
            mask = torch.randint(0, 2, (1, rows, 1), device=device).to(dtype)
            hidden = cudnn_linear_relu(value, weight_1, bias_1)
            assert hidden is not None
            output = cudnn_linear_mask(hidden, weight_2, bias_2, mask)
            assert output is not None
            expected_hidden = F.relu(torch.matmul(value, weight_1) + bias_1)
            expected_output = (torch.matmul(expected_hidden, weight_2) + bias_2) * mask
            torch.testing.assert_close(hidden, expected_hidden, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=2e-2)

    assert len(cudnn_graph_ops._LINEAR_RELU_CACHE) == len(row_counts)
    assert len(cudnn_graph_ops._LINEAR_MASK_CACHE) == len(row_counts)
    assert len(cudnn_graph_ops._DYNAMIC_LINEAR_RELU_CACHE) == 0
    assert len(cudnn_graph_ops._DYNAMIC_LINEAR_MASK_CACHE) == 0


def test_prepared_dynamic_linear_graphs_reuse_one_plan_across_row_counts() -> None:
    """The opt-in plans remain available and serve row counts they never saw."""
    input_dim, hidden_dim = 384, 768
    _clear_linear_plan_caches()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    linear_relu = prepare_cudnn_linear_relu(device, dtype, input_dim, hidden_dim)
    linear_mask = prepare_cudnn_linear_mask(device, dtype, hidden_dim, input_dim)
    assert linear_relu is not None
    assert linear_mask is not None
    assert len(cudnn_graph_ops._DYNAMIC_LINEAR_RELU_CACHE) == 1
    assert len(cudnn_graph_ops._DYNAMIC_LINEAR_MASK_CACHE) == 1

    weight_1, bias_1, weight_2, bias_2 = _released_pair_linear_operands(input_dim, hidden_dim, device, dtype)

    with torch.inference_mode():
        for rows in (64 * 64, 128 * 128):
            value = torch.randn(1, rows, input_dim, device=device, dtype=dtype)
            mask = torch.randint(0, 2, (1, rows, 1), device=device).to(dtype)
            hidden = linear_relu(value, weight_1, bias_1)
            assert hidden is not None
            output = linear_mask(hidden, weight_2, bias_2, mask)
            assert output is not None
            expected_hidden = F.relu(torch.matmul(value, weight_1) + bias_1)
            expected_output = (torch.matmul(expected_hidden, weight_2) + bias_2) * mask
            torch.testing.assert_close(hidden, expected_hidden, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=2e-2)

    assert len(cudnn_graph_ops._DYNAMIC_LINEAR_RELU_CACHE) == 1
    assert len(cudnn_graph_ops._DYNAMIC_LINEAR_MASK_CACHE) == 1
    assert len(cudnn_graph_ops._LINEAR_RELU_CACHE) == 0
    assert len(cudnn_graph_ops._LINEAR_MASK_CACHE) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cudnn_graph_primitives_match_rounded_torch_operations(dtype: torch.dtype) -> None:
    torch.manual_seed(41)
    rows, input_dim, output_dim = 29, 16, 32
    x = torch.randn(1, rows, input_dim, device="cuda", dtype=dtype)
    weight = torch.randn(1, input_dim, output_dim, device="cuda", dtype=dtype).mul_(input_dim**-0.5)
    bias = torch.randn(1, 1, output_dim, device="cuda", dtype=dtype).mul_(0.01)
    mask = torch.randint(0, 2, (1, rows, 1), device="cuda").to(dtype)

    with torch.inference_mode():
        hidden = cudnn_linear_relu(x, weight, bias)
        assert hidden is not None
        expected_hidden = F.relu(torch.matmul(x, weight) + bias)

        projected = cudnn_linear_mask(hidden, weight.transpose(-1, -2), bias[..., :input_dim], mask)
        assert projected is not None
        expected_projected = (torch.matmul(expected_hidden, weight.transpose(-1, -2)) + bias[..., :input_dim]) * mask
        old_value = torch.randn(1, rows, input_dim, device="cuda", dtype=dtype)
        projected_residual = cudnn_linear_residual(
            hidden,
            weight.transpose(-1, -2),
            bias[..., :input_dim],
            old_value,
        )
        assert projected_residual is not None
        expected_projected_residual = (
            torch.matmul(expected_hidden, weight.transpose(-1, -2)) + bias[..., :input_dim] + old_value
        )
        projected_mask_residual = cudnn_linear_mask_residual(
            hidden,
            weight.transpose(-1, -2),
            bias[..., :input_dim],
            mask,
            old_value,
        )
        assert projected_mask_residual is not None
        expected_projected_mask_residual = (
            torch.matmul(expected_hidden, weight.transpose(-1, -2)) + bias[..., :input_dim]
        ) * mask + old_value

        scale = torch.sigmoid(torch.randn_like(hidden))
        shift = torch.randn_like(hidden)
        modulated = cudnn_scale_shift_mask(hidden, scale, shift, mask)
        assert modulated is not None
        expected_modulated = (hidden * scale + shift) * mask

        first = torch.randn(1, 1, output_dim, device="cuda", dtype=dtype)
        second = torch.randn(1, rows, output_dim, device="cuda", dtype=dtype)
        added = cudnn_add_add_mask(hidden, first, second, mask)
        assert added is not None
        expected_added = (hidden + first + second) * mask

        residual = cudnn_add_mask(hidden, shift, mask)
        assert residual is not None
        expected_residual = (hidden + shift) * mask

    tolerance = {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 3e-3, "rtol": 3e-3}
    torch.testing.assert_close(hidden, expected_hidden, **tolerance)
    torch.testing.assert_close(projected, expected_projected, **tolerance)
    torch.testing.assert_close(projected_residual, expected_projected_residual, **tolerance)
    torch.testing.assert_close(projected_mask_residual, expected_projected_mask_residual, **tolerance)
    torch.testing.assert_close(modulated, expected_modulated, atol=0, rtol=0)
    torch.testing.assert_close(added, expected_added, atol=0, rtol=0)
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)


def test_cudnn_graph_eligibility_checks_flag_dtype_device_and_batch() -> None:
    value = torch.ones(1, 4, 8, device="cuda", dtype=torch.bfloat16)

    assert not can_use_cudnn_graph(value, enabled=False)
    assert can_use_cudnn_graph(value, enabled=True)
    assert can_use_cudnn_graph(value, enabled=True, require_batch_one=True)
    assert not can_use_cudnn_graph(value.expand(2, -1, -1), enabled=True, require_batch_one=True)
    assert not can_use_cudnn_graph(value.float(), enabled=True)
