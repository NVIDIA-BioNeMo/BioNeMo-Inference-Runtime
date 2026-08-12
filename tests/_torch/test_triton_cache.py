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

from tensorrt_bionemo.dsl_kernels.triton.fused_ln_proj_moveaxis_pad import FusedLNProjMoveaxisPad
from tensorrt_bionemo.dsl_kernels.triton.fused_swiglu import FusedSwiGLU
from tensorrt_bionemo.dsl_kernels.triton_cache import (
    _DRIVER_TRITON_OK,
    _SUPPORTED_TRITON_MAJOR_MINOR,
)


def test_supported_triton_driver_abi_is_36() -> None:
    assert _SUPPORTED_TRITON_MAJOR_MINOR == (3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_36_driver_fast_path_launches_fused_swiglu() -> None:
    pytest.importorskip("cuda.bindings.driver")
    assert _DRIVER_TRITON_OK

    d = 128
    torch.manual_seed(123)
    z = torch.randn(37, 2 * d, device="cuda", dtype=torch.bfloat16)
    op = FusedSwiGLU(d=d, three_way=False, dtype=z.dtype)
    kernel = op._kernels[z.dtype]
    assert kernel.driver is not None

    actual = op(z)
    torch.cuda.synchronize()
    expected = z[:, :d] * F.silu(z[:, d:])
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_ln_projection_driver_is_deterministic_in_cuda_graph() -> None:
    pytest.importorskip("cuda.bindings.driver")
    assert _DRIVER_TRITON_OK

    batch, i, j, d, heads = 1, 199, 199, 128, 4
    torch.manual_seed(123)
    z = torch.randn(batch, i, j, d, device="cuda", dtype=torch.bfloat16)
    ln_weight = torch.randn(d, device="cuda", dtype=torch.float32)
    ln_bias = torch.randn(d, device="cuda", dtype=torch.float32)
    proj_weight = torch.randn(heads, d, device="cuda", dtype=z.dtype)
    op = FusedLNProjMoveaxisPad(D=d, H=heads, dtype=z.dtype)
    signature = (z.dtype, ln_weight.dtype, ln_bias.dtype, proj_weight.dtype)
    assert op._kernels[signature].driver is not None

    for _ in range(3):
        op(z, ln_weight, ln_bias, proj_weight, multiple=8)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(z, ln_weight, ln_bias, proj_weight, multiple=8)
    graph.replay()
    eager_output = op(z, ln_weight, ln_bias, proj_weight, multiple=8)
    torch.cuda.synchronize()

    assert torch.equal(graph_output, eager_output)
    assert torch.count_nonzero(graph_output[..., j:]) == 0
