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

from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import _needs_int64, layer_norm_transpose


def _max_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Largest number of representable bf16 steps between two tensors."""
    as_int = actual.contiguous().view(torch.int16).to(torch.int32)
    ex_int = expected.contiguous().view(torch.int16).to(torch.int32)
    # Remap sign-magnitude onto a monotonic ordering so a subtraction counts
    # steps across zero as well as within a binade.
    as_int = torch.where(as_int < 0, -32768 - as_int, as_int)
    ex_int = torch.where(ex_int < 0, -32768 - ex_int, ex_int)
    return int((as_int - ex_int).abs().max())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_layer_norm_transpose_matches_torch(dtype: torch.dtype) -> None:
    torch.manual_seed(4)
    D, B, I, J = 196, 2, 17, 19
    x = torch.randn(D, B, I, J, device="cuda", dtype=dtype)
    weight = torch.randn(D, device="cuda", dtype=dtype)
    bias = torch.randn(D, device="cuda", dtype=dtype)

    with torch.inference_mode():
        actual = layer_norm_transpose(x, weight, bias, eps=1e-5, layout="dbij->bijd")
        expected = F.layer_norm(x.permute(1, 2, 3, 0), (D,), weight, bias, 1e-5)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("with_weight", "with_bias"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["no_affine", "weight_only", "bias_only", "weight_and_bias"],
)
def test_fused_layer_norm_supports_optional_affine(
    with_weight: bool,
    with_bias: bool,
) -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 17, 196, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(196, device="cuda", dtype=torch.bfloat16) if with_weight else None
    bias = torch.randn(196, device="cuda", dtype=torch.bfloat16) if with_bias else None

    with torch.inference_mode():
        actual = layer_norm_transpose(x, weight, bias, layout="bnd->bnd")
        expected = F.layer_norm(x, (196,), weight, bias, 1e-5)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_weight", [False, True], ids=["no_affine", "weight"])
@pytest.mark.parametrize("shape", [(2, 17, 196), (1, 512, 384), (3, 7, 768)])
def test_fused_layer_norm_transpose_supports_rms_norm(
    dtype: torch.dtype,
    with_weight: bool,
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(6)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype) if with_weight else None

    with torch.inference_mode():
        actual = layer_norm_transpose(x, weight, None, rms_norm=True, layout="bnd->bnd")
        expected = F.rms_norm(x, (shape[-1],), weight, eps=1e-5)

    tolerance = {
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
        torch.float32: 2e-5,
    }[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.is_contiguous()


@pytest.mark.parametrize("rms_norm", [False, True], ids=["layer_norm", "rms_norm"])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_norm_supports_fp32_affine_and_output_dtype(
    rms_norm: bool,
    out_dtype: torch.dtype,
) -> None:
    torch.manual_seed(9)
    x = torch.randn(2, 17, 196, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(196, device="cuda", dtype=torch.float32)
    bias = None if rms_norm else torch.randn(196, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        actual = layer_norm_transpose(
            x,
            weight,
            bias,
            rms_norm=rms_norm,
            layout="bnd->bnd",
            out_dtype=out_dtype,
        )
        if rms_norm:
            expected = F.rms_norm(x.float(), (196,), weight, eps=1e-5).to(out_dtype)
        else:
            expected = F.layer_norm(x.float(), (196,), weight, bias, 1e-5).to(out_dtype)

    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[out_dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.dtype == out_dtype


def test_fused_rms_norm_rejects_bias() -> None:
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="does not support an additive bias"):
        layer_norm_transpose(x, None, bias, rms_norm=True)


def test_fused_rms_norm_accepts_noncontiguous_input() -> None:
    torch.manual_seed(12)
    x = torch.randn(2, 17, 384, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    weight = torch.randn(384, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        actual = layer_norm_transpose(x, weight, None, rms_norm=True, layout="bnd->bnd")
        expected = F.rms_norm(x, (384,), weight, eps=1e-5)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert actual.is_contiguous()


def test_tiled_fused_rms_norm_matches_torch() -> None:
    torch.manual_seed(15)
    D = 5003
    x = torch.randn(3, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(D, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        actual = layer_norm_transpose(x, weight, None, rms_norm=True)
        expected = F.rms_norm(x, (D,), weight, eps=1e-5)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(0, 128), (2, 0)])
def test_empty_fused_norm_input(shape: tuple[int, ...]) -> None:
    x = torch.empty(*shape, device="cuda", dtype=torch.float32)
    weight = torch.empty(shape[-1], device="cuda", dtype=torch.float32)
    actual = layer_norm_transpose(x, weight, None, rms_norm=True)

    assert actual.shape == shape
    assert actual.dtype == x.dtype


def test_fused_norm_rejects_cpu_input() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        layer_norm_transpose(torch.randn(2, 128), None, None, rms_norm=True)


def test_fused_norm_rejects_unsupported_dtype() -> None:
    x = torch.randn(2, 128, device="cuda", dtype=torch.float64)
    with pytest.raises(ValueError, match="unsupported input dtype"):
        layer_norm_transpose(x, None, None, rms_norm=True)


def test_fused_norm_rejects_wrong_weight_shape() -> None:
    x = torch.randn(2, 128, device="cuda")
    weight = torch.randn(127, device="cuda")
    with pytest.raises(ValueError, match="weight must have shape"):
        layer_norm_transpose(x, weight, None, rms_norm=True)


@pytest.mark.parametrize("layout", ["dbij->bijd", "bijd->bijd", "bdij->bijd"])
@pytest.mark.parametrize("pad_multiple", [8, 16])
def test_pad_multiple_zero_extends_without_changing_values(layout: str, pad_multiple: int) -> None:
    """The pad tail is zero and the real channels are untouched."""
    torch.manual_seed(11)
    D, B, I, J = 196, 2, 9, 13
    shape = {
        "dbij->bijd": (D, B, I, J),
        "bijd->bijd": (B, I, J, D),
        "bdij->bijd": (B, D, I, J),
    }[layout]
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(D, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        unpadded = layer_norm_transpose(x, weight, bias, layout=layout)
        padded = layer_norm_transpose(x, weight, bias, layout=layout, pad_multiple=pad_multiple)

    d_padded = -(-D // pad_multiple) * pad_multiple
    assert padded.shape[-1] == d_padded
    # Statistics are taken over the true D, so padding must not move the head.
    # The bound is one representable step rather than exact equality: the pad
    # width is a compile-time constant, so it fixes the store's vector width and
    # can reorder the arithmetic that feeds it. That is a rounding artefact, and
    # both widths land the same distance from an fp32 reference. The failure this
    # guards against -- the zero tail entering the mean or variance -- would
    # scale every element in the row by about D / (D + pad), thousands of steps.
    assert _max_ulp_distance(padded[..., :D], unpadded) <= 1
    assert torch.all(padded[..., D:] == 0)


def test_pad_multiple_rejects_layouts_that_stride_the_channel_axis() -> None:
    x = torch.randn(1, 4, 4, 196, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(196, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(196, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous in D"):
        layer_norm_transpose(x, weight, bias, layout="bijd->bdij", pad_multiple=8)


def test_pad_multiple_must_divide_the_channel_tile() -> None:
    x = torch.randn(1, 4, 4, 196, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(196, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(196, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must divide"):
        layer_norm_transpose(x, weight, bias, layout="bijd->bijd", pad_multiple=48)


@pytest.mark.parametrize("rows", [1, 7, 132, 133, 1000, 4096, 20000])
@pytest.mark.parametrize("D", [128, 196, 384])
def test_row_counts_between_tiny_and_huge_still_launch(rows: int, D: int) -> None:
    """Every row count must yield a power-of-two row tile.

    The launcher caps the row tile so the grid still covers the SMs, and that
    bound is ``rows // sm_count`` -- an arbitrary quotient. Feeding it to
    ``tl.arange`` unrounded raises "range must be a power of 2", which only
    shows up between the tiny shapes the other tests use and the very large
    ones the models run.
    """
    torch.manual_seed(3)
    x = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(D, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        actual = layer_norm_transpose(x, weight, bias, eps=1e-5, layout="nd->nd")
        expected = F.layer_norm(x, (D,), weight, bias, 1e-5)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_int64_selection_uses_tiled_channel_extent() -> None:
    positions = 3072**2
    assert positions * 196 < 2**31 - 1
    assert positions * 256 >= 2**31 - 1
    assert _needs_int64(B=1, N=positions, D=196, tile_d=64)
    assert not _needs_int64(B=1, N=2048**2, D=196, tile_d=64)
