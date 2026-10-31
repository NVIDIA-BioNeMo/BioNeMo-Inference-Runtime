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
"""rms_linear_add against RMSNorm + Linear + residual in FP64."""

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir.dsl_kernels.triton.rms_linear_add import rms_linear_add

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(lead: tuple[int, ...], width: int, out_width: int, seed: int = 3):
    torch.manual_seed(seed)
    x = (torch.randn(*lead, width, device="cuda") * 2).bfloat16()
    residual = torch.randn(*lead, out_width, device="cuda").bfloat16()
    gamma = (1 + 0.2 * torch.randn(width, device="cuda")).bfloat16()
    weight = (torch.randn(out_width, width, device="cuda") * width**-0.5).bfloat16()
    return x, residual, gamma, weight


def _exact(x, residual, gamma, weight, eps):
    x64 = x.double()
    normed = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps) * gamma.double()
    return residual.double() + normed @ weight.double().T


@pytest.mark.parametrize(
    ("lead", "width", "out_width"),
    [((2, 37, 37), 384, 384), ((1, 300, 300), 384, 384), ((3, 5), 64, 96), ((7,), 96, 48)],
    ids=["pair-2x37", "large-rows", "narrow", "tile-fallback-widths"],
)
def test_rms_linear_add_is_closer_to_fp64_than_rounding_the_product(lead, width, out_width) -> None:
    """The fused form rounds once, after the residual; rounding the product first, as a separate GEMM does, is
    never more accurate."""
    eps = 1e-5
    x, residual, gamma, weight = _inputs(lead, width, out_width)
    row_sumsq = x.float().square().sum(-1)
    actual = rms_linear_add(residual, x, weight * gamma, row_sumsq, eps=eps)
    exact = _exact(x, residual, gamma, weight, eps)
    rstd = torch.rsqrt(row_sumsq / width + eps).unsqueeze(-1)
    two_step = (residual.float() + rstd * F.linear(x, weight * gamma).float()).bfloat16()

    assert actual.shape == residual.shape and actual.dtype == residual.dtype
    fused_error = (actual.double() - exact).abs().mean()
    assert fused_error <= (two_step.double() - exact).abs().mean() * 1.01
    torch.testing.assert_close(actual.double(), exact, rtol=2**-6, atol=2e-2)


def test_rms_linear_add_writes_over_its_input_or_its_residual() -> None:
    """An output sharing the residual is written in place; one sharing the projected rows gets a copy."""
    eps = 1e-5
    x, residual, gamma, weight = _inputs((2, 33, 33), 384, 384, seed=4)
    row_sumsq = x.float().square().sum(-1)
    expected = rms_linear_add(residual, x, weight * gamma, row_sumsq, eps=eps)

    over_residual = residual.clone()
    assert rms_linear_add(over_residual, x, weight * gamma, row_sumsq, eps=eps, out=over_residual) is (over_residual)
    assert torch.equal(over_residual, expected)
    over_x = x.clone()
    assert rms_linear_add(residual, over_x, weight * gamma, row_sumsq, eps=eps, out=over_x) is over_x
    assert torch.equal(over_x, expected)


def test_zero_rows_add_exactly_nothing_and_strided_inputs_take_the_pytorch_form() -> None:
    eps = 1e-5
    x, residual, gamma, weight = _inputs((4, 9), 128, 128, seed=5)
    zeros = torch.zeros_like(x)
    assert torch.equal(rms_linear_add(residual, zeros, weight, torch.zeros(4, 9, device="cuda"), eps=eps), residual)
    row_sumsq = x.float().square().sum(-1)
    strided = rms_linear_add(residual, x.mT.contiguous().mT, weight, row_sumsq, eps=eps)
    exact = _exact(x, residual, torch.ones_like(gamma), weight, eps)
    torch.testing.assert_close(strided.double(), exact, rtol=2**-6, atol=2e-2)


def test_a_cleared_flag_skips_the_call_and_a_set_one_does_not() -> None:
    """The device flag decides on the GPU: zero leaves ``out`` as it was, nonzero gives the unflagged result."""
    eps = 1e-5
    x, residual, gamma, weight = _inputs((2, 33, 33), 384, 384, seed=6)
    row_sumsq = x.float().square().sum(-1)
    expected = rms_linear_add(residual, x, weight * gamma, row_sumsq, eps=eps)
    on, off = (torch.tensor([flag], dtype=torch.int32, device="cuda") for flag in (1, 0))
    out = torch.full_like(residual, 7.0)
    rms_linear_add(residual, x, weight * gamma, row_sumsq, eps=eps, out=out, active=off)
    assert torch.equal(out, torch.full_like(residual, 7.0))
    assert torch.equal(rms_linear_add(residual, x, weight * gamma, row_sumsq, eps=eps, out=out, active=on), expected)
