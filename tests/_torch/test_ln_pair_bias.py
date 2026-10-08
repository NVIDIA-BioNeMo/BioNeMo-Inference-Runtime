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
"""``ln_pair_bias`` against PyTorch."""

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir.dsl_kernels.triton.ln_pair_bias import LNPairBias, ln_pair_bias

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8, reason="needs an SM80+ GPU"
)


def _reference(x, ln_weight, ln_bias, eps, weight, swap_ij, transposed, pad_multiple):
    """fp32 LayerNorm rounded once, then the heads in fp32 from the rounded rows."""
    rows = x.transpose(1, 2) if swap_ij else x
    normed = F.layer_norm(rows.float(), (x.shape[-1],), ln_weight.float(), ln_bias.float(), eps).to(x.dtype)
    bias = F.linear(normed.float(), weight.float())
    bias = (bias.transpose(1, 2) if transposed else bias).permute(0, 3, 1, 2)
    pad = -bias.shape[-1] % pad_multiple if pad_multiple > 0 else 0
    return normed, F.pad(bias, (0, pad)).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("channels", "heads"), [(64, 2), (128, 4), (256, 8), (256, 16)])
@pytest.mark.parametrize("swap_ij", [False, True], ids=["starting", "ending"])
@pytest.mark.parametrize("transposed", [False, True], ids=["bias", "transposed-bias"])
@pytest.mark.parametrize(("tokens", "pad_multiple"), [(37, -1), (37, 8), (64, 8)])
def test_matches_pytorch(dtype, channels, heads, swap_ij, transposed, tokens, pad_multiple) -> None:
    """Both outputs match PyTorch; padded keys are zero."""
    torch.manual_seed(0)
    x = torch.randn(2, tokens, tokens + 3, channels, device="cuda", dtype=dtype)
    ln_weight = torch.randn(channels, device="cuda", dtype=dtype) * 0.2 + 1
    ln_bias = torch.randn(channels, device="cuda", dtype=dtype) * 0.2
    weight = torch.randn(heads, channels, device="cuda", dtype=dtype) * channels**-0.5
    normed, bias = ln_pair_bias(
        x, ln_weight, ln_bias, 1e-5, weight, swap_ij=swap_ij, transposed=transposed, pad_multiple=pad_multiple
    )
    expected_normed, expected_bias = _reference(x, ln_weight, ln_bias, 1e-5, weight, swap_ij, transposed, pad_multiple)
    assert normed.shape == expected_normed.shape and bias.shape == expected_bias.shape
    assert normed.is_contiguous() and bias.is_contiguous()
    torch.testing.assert_close(normed, expected_normed, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bias, expected_bias, atol=2e-2, rtol=2e-2)
    keys = x.shape[1] if swap_ij != transposed else x.shape[2]
    assert not bias[..., keys:].any()


def test_non_contiguous_pair_reads_like_a_contiguous_one() -> None:
    """A strided pair gives its contiguous copy's output."""
    torch.manual_seed(1)
    x = torch.randn(1, 40, 40, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    ln_weight = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.2 + 1
    ln_bias = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16) * 0.0625
    expected = ln_pair_bias(x.contiguous(), ln_weight, ln_bias, 1e-5, weight, pad_multiple=8)
    actual = ln_pair_bias(x, ln_weight, ln_bias, 1e-5, weight, pad_multiple=8)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.equal(result, reference)


def test_shares_one_compilation_and_replays_in_a_cuda_graph() -> None:
    """Ops of one configuration share a kernel; a captured call replays on new inputs."""
    op = LNPairBias(256, 8, swap_ij=True)
    assert op._kernel is not None and op._kernel is LNPairBias(256, 8, swap_ij=True)._kernel
    torch.manual_seed(2)
    x = torch.randn(1, 123, 123, 256, device="cuda", dtype=torch.bfloat16)
    ln_weight = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.2 + 1
    ln_bias = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.2
    weight = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16) * 0.0625
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(x, ln_weight, ln_bias, 1e-5, weight, pad_multiple=8)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, ln_weight, ln_bias, 1e-5, weight, pad_multiple=8)
    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = op(x, ln_weight, ln_bias, 1e-5, weight, pad_multiple=8)
    for result, reference in zip(captured, expected, strict=True):
        assert torch.equal(result, reference)


@pytest.mark.parametrize("case", ["fp32", "width-96", "width-512", "17-heads"])
def test_no_kernel_outside_its_envelope(case) -> None:
    """Configurations without a kernel return ``None``."""
    channels = {"width-96": 96, "width-512": 512}.get(case, 256)
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    x = torch.randn(1, 8, 8, channels, device="cuda", dtype=dtype)
    ln_weight = torch.ones(channels, device="cuda", dtype=dtype)
    ln_bias = torch.zeros(channels, device="cuda", dtype=dtype)
    weight = torch.zeros(17 if case == "17-heads" else 8, channels, device="cuda", dtype=dtype)
    assert ln_pair_bias(x, ln_weight, ln_bias, 1e-5, weight) is None
