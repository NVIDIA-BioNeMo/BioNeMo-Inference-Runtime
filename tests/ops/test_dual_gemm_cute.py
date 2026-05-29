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
"""Tests for CuTe DSL dual GEMM x0_x1: sigmoid(X0 @ W0.T [+ bias0]) * (X1 @ W1.T [+ bias1])."""

from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from tests._torch import skip_if_no_cutedsl
from tensorrt_bionemo._torch.custom_ops.dual_gemm_x0_x1 import \
    DualGemmX0X1CuTe


def _ref_x0_x1_dual_gemm(
    X0: torch.Tensor,
    X1: torch.Tensor,
    W0: torch.Tensor,
    W1: torch.Tensor,
    bias0: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks."""
    d0 = torch.nn.functional.linear(X0.float(), W0.float(),
                                     bias0.float() if bias0 is not None else None)
    d1 = torch.nn.functional.linear(X1.float(), W1.float(),
                                     bias1.float() if bias1 is not None else None)
    return (d0.sigmoid() * d1).to(X0.dtype)


@dataclass
class Scenario:
    K: int = 128
    N: int = 128
    seq_lens: list[int] = field(
        default_factory=lambda: [100, 123, 512, 1023, 1024])
    has_bias: bool = False
    dtype: torch.dtype = torch.bfloat16
    atol: float = 1e-2
    rtol: float = 1e-2


@pytest.mark.parametrize("sc", [
    Scenario(N=128, K=128, seq_lens=[100], dtype=torch.bfloat16),
    Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True),
    Scenario(N=128, K=128, dtype=torch.float16),
    Scenario(N=128, K=128, dtype=torch.float16, has_bias=True),
],
    ids=[
        "sc_N128_K128_b0_bf16",
        "sc_N128_K128_b1_bf16",
        "sc_N128_K128_b0_fp16",
        "sc_N128_K128_b1_fp16",
    ])
def test_x0_x1_dual_gemm(sc: Scenario):
    """Test CuTe DSL dual GEMM x0_x1 against fp32 reference."""
    skip_if_no_cutedsl()
    torch.manual_seed(42)

    cute_op = DualGemmX0X1CuTe()

    W0 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    W1 = torch.randn(sc.N, sc.K, dtype=sc.dtype, device="cuda")
    bias0 = torch.randn(sc.N, dtype=sc.dtype, device="cuda") if sc.has_bias else None
    bias1 = torch.randn(sc.N, dtype=sc.dtype, device="cuda") if sc.has_bias else None

    for seq_len in sc.seq_lens:
        X0 = torch.randn(1, seq_len, seq_len, sc.K,
                          device="cuda").contiguous().to(sc.dtype)
        X1 = torch.randn(1, seq_len, seq_len, sc.K,
                          device="cuda").contiguous().to(sc.dtype)

        ref = _ref_x0_x1_dual_gemm(X0, X1, W0, W1, bias0, bias1)
        out = cute_op(X0, X1, W0, W1, bias0, bias1)

        torch.testing.assert_close(
            out,
            ref,
            atol=sc.atol,
            rtol=sc.rtol,
            msg=lambda m: f"seq_len={seq_len}, has_bias={sc.has_bias}: {m}",
        )
