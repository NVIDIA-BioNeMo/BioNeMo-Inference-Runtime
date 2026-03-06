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

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn
from tensorrt_llm_lite._utils import get_sm_version

from tensorrt_bionemo.ops import x0_x1_dual_gemm, x_x_dual_gemm


@dataclass
class X_X_Scenario:
    K: int = 128
    N: int = 256
    seq_lens: list[int] = field(
        default_factory=lambda: [100, 123, 186, 512, 1023, 1024])
    has_bias: bool = False
    has_mask: bool = False
    dtype: torch.dtype = torch.bfloat16


def ref_torch_x_x_dual_gemm(X: torch.Tensor,
                            linear0: torch.nn.Linear,
                            linear1: torch.nn.Linear,
                            mask: torch.Tensor = None) -> torch.Tensor:
    d0 = linear0(X)
    if mask is not None:
        return d0.sigmoid() * linear1(X) * mask.unsqueeze(-1)
    else:
        return d0.sigmoid() * linear1(X)


def create_linear_layers(K: int, N: int, dtype=torch.bfloat16, has_bias=False):
    linear0 = nn.Linear(K, N, dtype=torch.float32, bias=has_bias).cuda()
    torch.nn.init.xavier_uniform_(linear0.weight)
    if has_bias:
        torch.nn.init.uniform_(linear0.bias)

    linear1 = nn.Linear(K, N, dtype=torch.float32, bias=has_bias).cuda()
    torch.nn.init.xavier_uniform_(linear1.weight)
    if has_bias:
        torch.nn.init.uniform_(linear1.bias)

    return linear0.to(dtype), linear1.to(dtype)


@pytest.mark.parametrize("sc", [
    X_X_Scenario(N=256, K=128, dtype=torch.bfloat16),
    X_X_Scenario(N=256, K=128, dtype=torch.bfloat16, has_bias=True),
    X_X_Scenario(N=256, K=128, dtype=torch.bfloat16, has_mask=True),
    X_X_Scenario(
        N=256, K=128, dtype=torch.bfloat16, has_bias=True, has_mask=True),
    X_X_Scenario(N=256, K=128, dtype=torch.float16),
    X_X_Scenario(N=256, K=128, dtype=torch.float16, has_bias=True),
    X_X_Scenario(N=256, K=128, dtype=torch.float16, has_mask=True),
    X_X_Scenario(
        N=256, K=128, dtype=torch.float16, has_bias=True, has_mask=True),
    X_X_Scenario(N=128, K=128, dtype=torch.bfloat16),
    X_X_Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True),
    X_X_Scenario(N=128, K=128, dtype=torch.bfloat16, has_mask=True),
    X_X_Scenario(
        N=128, K=128, dtype=torch.bfloat16, has_bias=True, has_mask=True),
    X_X_Scenario(N=128, K=128, dtype=torch.float16),
    X_X_Scenario(N=128, K=128, dtype=torch.float16, has_bias=True),
    X_X_Scenario(N=128, K=128, dtype=torch.float16, has_mask=True),
    X_X_Scenario(
        N=128, K=128, dtype=torch.float16, has_bias=True, has_mask=True),
],
                         ids=[
                             "sc_N256_K128_b0_m0_bf16",
                             "sc_N256_K128_b1_m0_bf16",
                             "sc_N256_K128_b0_m1_bf16",
                             "sc_N256_K128_b1_m1_bf16",
                             "sc_N256_K128_b0_m0_fp16",
                             "sc_N256_K128_b1_m0_fp16",
                             "sc_N256_K128_b0_m1_fp16",
                             "sc_N256_K128_b1_m1_fp16",
                             "sc_N128_K128_b0_m0_bf16",
                             "sc_N128_K128_b1_m0_bf16",
                             "sc_N128_K128_b0_m1_bf16",
                             "sc_N128_K128_b1_m1_bf16",
                             "sc_N128_K128_b0_m0_fp16",
                             "sc_N128_K128_b1_m0_fp16",
                             "sc_N128_K128_b0_m1_fp16",
                             "sc_N128_K128_b1_m1_fp16",
                         ])
def test_x_x_dual_gemm(sc: X_X_Scenario):
    sm = get_sm_version()
    if sm < 80 or sm >= 90:
        pytest.skip("x_x_dual_gemm is not supported on SM < 80 or >= 90")
    linear0, linear1 = create_linear_layers(sc.K, sc.N, sc.dtype, sc.has_bias)
    for seq_len in sc.seq_lens:
        X = torch.randn(1, seq_len, seq_len, sc.K,
                        device="cuda").contiguous().to(sc.dtype)
        if sc.has_mask:
            mask = torch.randint(0,
                                 2, (1, seq_len, seq_len),
                                 device="cuda",
                                 dtype=torch.float32).to(sc.dtype)
        else:
            mask = None
        output = x_x_dual_gemm(X, linear0.weight, linear1.weight, linear0.bias,
                               linear1.bias, mask)
        ref_output = ref_torch_x_x_dual_gemm(X, linear0, linear1, mask)
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


@dataclass
class X0_X1_Scenario:
    K: int = 128
    N: int = 128
    seq_lens: list[int] = field(
        default_factory=lambda: [100, 123, 512, 1023, 1024])
    has_bias: bool = False
    dtype: torch.dtype = torch.bfloat16


def ref_torch_x0_x1_dual_gemm(X0: torch.Tensor, X1: torch.Tensor,
                              linear0: torch.nn.Linear,
                              linear1: torch.nn.Linear) -> torch.Tensor:
    d0 = linear0(X0)
    d1 = linear1(X1)
    return d0.sigmoid() * d1


@pytest.mark.parametrize(
    "sc",
    [
        X0_X1_Scenario(N=128, K=128, dtype=torch.bfloat16),
        X0_X1_Scenario(N=128, K=128, dtype=torch.bfloat16, has_bias=True),
        X0_X1_Scenario(N=128, K=128, dtype=torch.float16),
        X0_X1_Scenario(N=128, K=128, dtype=torch.float16, has_bias=True),
    ],
    ids=[
        "sc_N128_K128_b0_bf16",
        "sc_N128_K128_b1_bf16",
        "sc_N128_K128_b0_fp16",
        "sc_N128_K128_b1_fp16",
    ],
)
def test_x0_x1_dual_gemm(sc: X0_X1_Scenario):
    sm = get_sm_version()
    if sm < 80 or sm >= 90:
        pytest.skip("x0_x1_dual_gemm is not supported on SM < 80 or >= 90")
    linear0, linear1 = create_linear_layers(sc.K, sc.N, sc.dtype, sc.has_bias)
    for seq_len in sc.seq_lens:
        X0 = torch.randn(1, seq_len, seq_len, sc.K,
                         device="cuda").contiguous().to(sc.dtype)
        X1 = torch.randn(1, seq_len, seq_len, sc.K,
                         device="cuda").contiguous().to(sc.dtype)
        output = x0_x1_dual_gemm(X0, X1, linear0.weight, linear1.weight,
                                 linear0.bias, linear1.bias)
        ref_output = ref_torch_x0_x1_dual_gemm(X0, X1, linear0, linear1)
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
