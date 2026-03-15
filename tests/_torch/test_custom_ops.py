# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.custom_ops.dual_gemm import get_dual_gemm_op


def create_linear_layers(K: int = 128,
                         N: int = 256,
                         dtype=torch.bfloat16,
                         has_bias=False):
    linear0 = nn.Linear(K, N, dtype=torch.float32, bias=has_bias).cuda()
    torch.nn.init.xavier_uniform_(linear0.weight)
    if has_bias:
        torch.nn.init.uniform_(linear0.bias)

    linear1 = nn.Linear(K, N, dtype=torch.float32, bias=has_bias).cuda()
    torch.nn.init.xavier_uniform_(linear1.weight)
    if has_bias:
        torch.nn.init.uniform_(linear1.bias)

    return linear0.to(dtype), linear1.to(dtype)


def test_fused_sigmoid_gated_dual_gemm():
    torch.manual_seed(42)
    seq_len = 256
    dtype = torch.bfloat16
    K = 128
    N = 256
    linear0, linear1 = create_linear_layers(K=K,
                                            N=N,
                                            dtype=dtype,
                                            has_bias=False)
    x = torch.randn(1, seq_len, seq_len, K,
                    device="cuda").contiguous().to(dtype)
    mask = torch.randint(0,
                         2, (1, seq_len, seq_len),
                         device="cuda",
                         dtype=torch.float32).to(dtype)
    ref_x = linear0(x).sigmoid() * linear1(x)
    ref_x = ref_x * mask.unsqueeze(-1)

    op = get_dual_gemm_op(dtype,
                          transpose_out=False,
                          dual_gemm_type="x_x",
                          N=N,
                          K=K)
    out = op(x, linear0.weight, linear1.weight, linear0.bias, linear1.bias,
             mask)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)

    x = torch.randn(1, seq_len, seq_len, K,
                    device="cuda").contiguous().to(dtype)
    linear0, linear1 = create_linear_layers(K=K,
                                            N=N,
                                            dtype=dtype,
                                            has_bias=True)

    ref_x = linear0(x).sigmoid() * linear1(x)
    ref_x = ref_x * mask.unsqueeze(-1)

    op = get_dual_gemm_op(dtype,
                          transpose_out=False,
                          dual_gemm_type="x_x",
                          N=N,
                          K=K)
    out = op(x, linear0.weight, linear1.weight, linear0.bias, linear1.bias,
             mask)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)


def test_fused_sigmoid_gated_dual_gemm_dual_x():
    torch.manual_seed(42)
    seq_len = 256
    dtype = torch.bfloat16
    K = 128
    N = 256
    linear0, linear1 = create_linear_layers(K=K,
                                            N=N,
                                            dtype=dtype,
                                            has_bias=False)
    x0 = torch.randn(1, seq_len, seq_len, K,
                     device="cuda").contiguous().to(dtype)
    x1 = torch.randn(1, seq_len, seq_len, K,
                     device="cuda").contiguous().to(dtype)
    ref_x = linear0(x0).sigmoid() * linear1(x1)

    op = get_dual_gemm_op(dtype,
                          transpose_out=False,
                          dual_gemm_type="x0_x1",
                          N=N,
                          K=K)
    out = op(x0, x1, linear0.weight, linear1.weight, linear0.bias,
             linear1.bias, None)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)

    linear0, linear1 = create_linear_layers(K=K,
                                            N=N,
                                            dtype=dtype,
                                            has_bias=True)
    ref_x = linear0(x0).sigmoid() * linear1(x1)
    op = get_dual_gemm_op(dtype,
                          transpose_out=False,
                          dual_gemm_type="x0_x1",
                          N=N,
                          K=K)
    out = op(x0, x1, linear0.weight, linear1.weight, linear0.bias,
             linear1.bias, None)
    torch.testing.assert_close(out, ref_x, atol=5e-1, rtol=1e-2)
