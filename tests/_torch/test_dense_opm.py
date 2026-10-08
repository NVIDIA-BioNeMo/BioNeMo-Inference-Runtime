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
"""Qualify the Triton OPM fallback: arithmetic, replay, launch paths, and routing."""

import pytest
import torch

from bionemo_ir._torch.custom_ops.outer_product_mean import ops
from bionemo_ir._torch.custom_ops.outer_product_mean.ops import (
    _invoke_cute_opm,
    _invoke_triton_opm,
    _invoke_vanilla_opm,
    get_outer_product_mean_op,
)
from bionemo_ir._torch.layers.outer_product_mean import OuterProductMean
from bionemo_ir.dsl_kernels.triton import dense_outer_product as triton_opm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0), reason="Triton OPM needs SM80"
)

# (I, J, S, B): tails on every tile axis, I != J, the 256-row chunk boundary, and the batch loop.
SHAPES = [
    (1, 1, 1, 1),
    (17, 19, 31, 1),
    (129, 130, 33, 1),
    (257, 257, 100, 1),
    (515, 333, 257, 1),
    (520, 520, 512, 1),
    (300, 300, 65, 2),
]


def inputs(
    rows: int, cols: int, sequences: int, batch: int = 1, c_z: int = 128, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Per-shape data, so a stale buffer from another shape cannot pass for a result.
    torch.manual_seed(rows + 3 * cols + 7 * sequences + 11 * batch + c_z)
    a = torch.randn(batch, sequences, rows, 32, dtype=dtype, device="cuda")
    b = torch.randn(batch, sequences, cols, 32, dtype=dtype, device="cuda")
    weight = torch.randn(c_z, 1024, dtype=a.dtype, device=a.device) * 0.05
    bias = torch.randn(c_z, dtype=a.dtype, device=a.device) * 0.1
    norm = torch.randint(sequences // 2 + 1, sequences + 2, (batch, rows, cols), device=a.device).float()
    if rows > 1 and cols > 1:
        # Token 0 is fully masked: zero operands over a tiny count. Tail tokens keep data.
        a[:, :, 0] = 0
        b[:, :, 0] = 0
        norm[:, 0] = 1e-3
        norm[:, :, 0] = 1e-3
    return a, b, norm, weight, bias


def assert_matches_reference(
    actual: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    norm: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    before: bool,
) -> None:
    expected = _invoke_vanilla_opm(a, b, norm, weight, bias, before).float()
    assert actual.shape == expected.shape and actual.dtype == a.dtype
    # Elementwise against the median scale: a wrong tile is off by a whole output, BF16 by a few ulps.
    atol = 0.02 * expected.abs().median().item()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=atol)


@pytest.mark.parametrize("rows,cols,sequences,batch", SHAPES)
@pytest.mark.parametrize("c_z", [128, 256])
@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_arithmetic(
    rows: int, cols: int, sequences: int, batch: int, c_z: int, before: bool, has_bias: bool, dtype: torch.dtype
) -> None:
    a, b, norm, weight, bias = inputs(rows, cols, sequences, batch, c_z, dtype)
    bias = bias if has_bias else None
    actual = _invoke_triton_opm(a, b, norm, weight, bias, before)
    assert_matches_reference(actual, a, b, norm, weight, bias, before)


def test_wide_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 64-bit sequence offsets match the 32-bit ones."""
    a, b, norm, weight, bias = inputs(257, 130, 100)
    narrow = _invoke_triton_opm(a, b, norm, weight, bias)
    monkeypatch.setattr(triton_opm, "_INDEX_LIMIT", 1)
    wide = _invoke_triton_opm(a, b, norm, weight, bias)
    torch.testing.assert_close(wide, narrow, rtol=0, atol=0)


@pytest.mark.parametrize("c_z", [128, 256])
@pytest.mark.parametrize("before", [True, False])
def test_replay(c_z: int, before: bool) -> None:
    a, b, norm, weight, bias = inputs(520, 515, 300, c_z=c_z)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _invoke_triton_opm(a, b, norm, weight, bias, before)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _invoke_triton_opm(a, b, norm, weight, bias, before)
    for scale in [-1, 2]:
        a.mul_(scale)
        b[:, :, :4] = 0
        norm.add_(1)
        weight.neg_()
        bias.add_(0.01)
        graph.replay()
        assert_matches_reference(actual, a, b, norm, weight, bias, before)


@pytest.mark.parametrize("use_driver", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_cached_launch(monkeypatch: pytest.MonkeyPatch, use_driver: bool, has_bias: bool) -> None:
    from bionemo_ir.dsl_kernels.triton_cache import _DRIVER_TRITON_OK

    a, b, norm, weight, bias = inputs(520, 520, 288)
    bias = bias if has_bias else None
    kernels = triton_opm._cached_kernels(a.device.index, a.dtype, 128, has_bias, False, False)
    assert kernels is triton_opm._cached_kernels(a.device.index, a.dtype, 128, has_bias, False, False)
    for kernel in (kernels.contract, kernels.project):
        if use_driver:
            if not _DRIVER_TRITON_OK:
                pytest.skip("Unsupported Triton driver ABI")
            assert kernel.driver is not None
        else:
            monkeypatch.setattr(kernel, "_driver", None)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        actual = _invoke_triton_opm(a, b, norm, weight, bias, False)
    torch.cuda.current_stream().wait_stream(side)
    assert_matches_reference(actual, a, b, norm, weight, bias, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _invoke_triton_opm(a, b, norm, weight, bias, False)
    a.neg_()
    graph.replay()
    assert_matches_reference(actual, a, b, norm, weight, bias, False)


def test_unaligned_operands() -> None:
    tensors = inputs(520, 517, 288)
    a, b, norm, weight, bias = [
        torch.cat((tensor.new_zeros(1), tensor.flatten()))[1:].view_as(tensor) for tensor in tensors
    ]
    assert all(tensor.data_ptr() % 16 for tensor in (a, b, norm, weight, bias))
    actual = _invoke_triton_opm(a, b, norm, weight, bias)
    assert_matches_reference(actual, *tensors, True)


def test_strided_operands() -> None:
    a, b, norm, weight, bias = inputs(520, 520, 512)
    strided = (
        a.transpose(1, 2).contiguous().transpose(1, 2),
        b.transpose(1, 2).contiguous().transpose(1, 2),
        norm.transpose(1, 2).contiguous().transpose(1, 2),
        weight.t().contiguous().t(),
        bias,
    )
    actual = _invoke_triton_opm(*strided)
    assert_matches_reference(actual, a, b, norm, weight, bias, True)


@pytest.mark.parametrize("kind", ["fp32", "mixed", "c_z", "channels", "sequences", "bias", "mask", "device"])
def test_rejects_unsupported(kind: str) -> None:
    a, b, norm, weight, bias = inputs(17, 19, 31)
    if kind == "fp32":
        a, b, weight, bias = [x.float() for x in (a, b, weight, bias)]
    elif kind == "mixed":
        weight = weight.half()
    elif kind == "c_z":
        weight, bias = weight[:64], bias[:64]
    elif kind == "channels":
        a = a[..., :16]
    elif kind == "sequences":
        b = b[:, :30]
    elif kind == "bias":
        bias = bias[:64]
    elif kind == "mask":
        norm = norm[:, :, :18]
    elif kind == "device":
        norm = norm.cpu()
    with pytest.raises(ValueError, match="dense_outer_product takes"):
        _invoke_triton_opm(a, b, norm, weight, bias)


# Native CuTe kernels by SM. Every other SM80+ SKU, Blackwell included, runs Triton first.
NATIVE_CUTE = {
    80: (torch.float16, torch.bfloat16),
    86: (torch.float16, torch.bfloat16),
    89: (torch.float16, torch.bfloat16),
    90: (torch.bfloat16,),
}


@pytest.mark.parametrize("sm", [75, 80, 86, 87, 89, 90, 100, 103, 110, 120, 121])
@pytest.mark.parametrize("c_z", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_selector(monkeypatch: pytest.MonkeyPatch, sm: int, c_z: int, dtype: torch.dtype) -> None:
    """CuTe runs where it has a native kernel; Triton goes first on every other SM80+ SKU."""
    monkeypatch.setattr(ops, "get_sm_version", lambda: sm)
    if dtype in NATIVE_CUTE.get(sm, ()):
        expected = _invoke_cute_opm
    else:
        expected = _invoke_triton_opm if sm >= 80 else _invoke_vanilla_opm
    assert get_outer_product_mean_op(dtype, 32, 32, c_z) is expected
    for other_dtype, channels, other_c_z in ((torch.float32, 32, 128), (dtype, 64, 128), (dtype, 32, 192)):
        assert get_outer_product_mean_op(other_dtype, channels, channels, other_c_z) is _invoke_vanilla_opm


@pytest.mark.parametrize("c_z", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_layer_dispatch(monkeypatch: pytest.MonkeyPatch, c_z: int, dtype: torch.dtype) -> None:
    monkeypatch.setattr(ops, "get_sm_version", lambda: 120)
    layer = OuterProductMean(16, 32, c_z, dtype=dtype).cuda().eval()
    assert layer._opm_op is _invoke_triton_opm and layer._opm_eligible
    torch.manual_seed(5)
    layer.load_state_dict({name: torch.randn_like(value) * 0.1 for name, value in layer.state_dict().items()})
    m = torch.randn(1, 97, 263, 16, device="cuda", dtype=dtype)
    mask = (torch.rand(m.shape[:-1], device=m.device) < 0.9).to(dtype)
    actual = layer(m, mask)
    layer._opm_eligible = False
    expected = layer(m, mask)
    rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert actual.shape == (1, 263, 263, c_z) and rel_l2 < 1e-2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_device_cache() -> None:
    for device in (1, 0, 1):
        with torch.cuda.device(device):
            a, b, norm, weight, bias = inputs(512, 512, 256)
            with torch.cuda.device(1 - device):
                actual = _invoke_triton_opm(a, b, norm, weight, bias)
                assert torch.cuda.current_device() == 1 - device
            assert_matches_reference(actual, a, b, norm, weight, bias, True)
