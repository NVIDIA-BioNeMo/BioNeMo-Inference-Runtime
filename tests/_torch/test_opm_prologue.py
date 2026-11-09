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
"""The fused outer-product-mean prologue against the eager LayerNorm, projection, mask and split."""

import pytest
import torch

from bionemo_ir._torch.custom_ops.outer_product_mean import outer_product_mean_a_chunk, outer_product_mean_a_chunks
from bionemo_ir._torch.custom_ops.outer_product_mean.cutedsl import _relayout_a
from bionemo_ir._torch.layers.outer_product_mean import OuterProductMean
from bionemo_ir.dsl_kernels.triton.opm_prologue import OpmPrologue

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the OPM prologue needs CUDA")


def _eager(m, mask, ln_w, ln_b, eps, weight, bias):
    x = torch.nn.functional.layer_norm(m, (m.shape[-1],), ln_w, ln_b, eps)
    ab = torch.nn.functional.linear(x, weight, bias)
    a, b = ab.split(weight.shape[0] // 2, dim=-1)
    return a * mask.unsqueeze(-1), b * mask.unsqueeze(-1)


def _close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # The LayerNorm reduction order differs from PyTorch's, so a few values land one rounding step away.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1.6e-2, atol=2e-2)
    assert (actual != expected).float().mean().item() < 1e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("shape", [(1, 1, 1, 64), (2, 37, 65, 64), (1, 173, 300, 64), (1, 5, 129, 32)])
@pytest.mark.parametrize("chunk", [None, 4, 8])
@pytest.mark.parametrize("has_bias", [True, False])
def test_prologue_matches_eager(dtype: torch.dtype, shape: tuple[int, ...], chunk: int | None, has_bias: bool) -> None:
    torch.manual_seed(0)
    B, S, I, K = shape
    m = torch.randn(shape, device="cuda", dtype=dtype)
    mask = (torch.rand(B, S, I, device="cuda") > 0.2).to(dtype)
    ln_w, ln_b = (torch.randn(K, device="cuda", dtype=dtype) for _ in range(2))
    weight = torch.randn(64, K, device="cuda", dtype=dtype) / K**0.5
    bias = torch.randn(64, device="cuda", dtype=dtype) if has_bias else None
    a, b = OpmPrologue()(m, mask, ln_w, ln_b, 1e-5, weight, bias, chunk=chunk)
    a_ref, b_ref = _eager(m, mask, ln_w, ln_b, 1e-5, weight, bias)
    _close(b, b_ref)
    if chunk is None:
        _close(a, a_ref)
    else:
        expected = _relayout_a(a_ref, chunk)
        assert a.shape == expected.shape and a.stride() == expected.stride()
        # Pad tokens are never written; compare the real ones.
        T = 128 // chunk
        real = torch.arange(a.shape[2] * T, device="cuda").view(a.shape[2], T) < I
        _close(a.permute(0, 1, 3, 5, 2, 4)[..., real], expected.permute(0, 1, 3, 5, 2, 4)[..., real])


def test_prologue_rejects_mismatched_mask() -> None:
    m = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    assert not OpmPrologue.supports(m, torch.ones(1, 4, 8, device="cuda"), 32)
    assert not OpmPrologue.supports(m.float(), torch.ones(1, 4, 8, device="cuda"), 32)
    assert OpmPrologue.supports(m, torch.ones(1, 4, 8, device="cuda", dtype=torch.bfloat16), 32)


@pytest.mark.parametrize(
    "c_out,norm_before,S,N",
    [(256, False, 173, 200), (128, False, 134, 130), (128, True, 70, 257), (256, False, 1, 8), (128, True, 0, 16)],
)
def test_layer_fused_path_matches_unfused(c_out: int, norm_before: bool, S: int, N: int) -> None:
    torch.manual_seed(0)
    kwargs = {} if norm_before else {"norm_mask_by_eps": True, "norm_before_output": False}
    layer = OuterProductMean(64, 32, c_out, dtype=torch.bfloat16, **kwargs).cuda()
    if not layer._opm_eligible:
        pytest.skip("no fused OPM backend on this device")
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0, 0.1)
        m = torch.randn(1, S, N, 64, device="cuda", dtype=torch.bfloat16)
        mask = (torch.rand(1, S, N, device="cuda") > 0.1).to(torch.bfloat16)
        fused = layer(m, mask)
        # The unfused route: eager LayerNorm, projection and mask, then the same fused op.
        a, b = _eager(
            m,
            mask,
            layer.norm.weight,
            layer.norm.bias,
            layer.norm.eps,
            layer.fused_proj_a_b.weight,
            layer.fused_proj_a_b.bias,
        )
        num_mask = layer._compute_num_mask(mask.unsqueeze(-1), dtype=torch.float32).squeeze(-1)
        unfused = layer._opm_op(a, b, num_mask, layer.proj_o.weight, layer.proj_o.bias, norm_before=norm_before)
    assert fused.shape == (1, N, N, c_out)
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused.float(), unfused.float(), rtol=3e-2, atol=3e-2)


def test_layer_compiles_every_prologue_layout_at_init(monkeypatch) -> None:
    """The layer compiles the prologue for every layout of ``a`` its backend can take, so forwards whose depths route
    to different layouts (on SM100 a resident-b tile, a streaming tile and Triton) compile nothing."""
    monkeypatch.setattr(OpmPrologue, "_global_cache", {})
    layer = OuterProductMean(64, 32, 128, dtype=torch.bfloat16).cuda()
    if not layer._opm_eligible:
        pytest.skip("no fused OPM backend on this device")

    def compile_forbidden(*args, **kwargs):
        raise AssertionError("the prologue compiled during a forward")

    monkeypatch.setattr(OpmPrologue, "compile_for_dtypes", compile_forbidden)
    prepared = {None, *outer_product_mean_a_chunks(128, torch.bfloat16)}
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0, 0.1)
        for S in (40, 3000, 8193):
            assert outer_product_mean_a_chunk(S, 16, 16, 128, True, True, torch.bfloat16) in prepared, S
            m = torch.randn(1, S, 16, 64, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(1, S, 16, device="cuda", dtype=torch.bfloat16)
            assert torch.isfinite(layer(m, mask)).all()
