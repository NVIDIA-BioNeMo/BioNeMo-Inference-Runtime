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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Numerical tests for the CuTeDSL backend of AdaLN.

Compares the default kernel-backed ``AdaLN`` against the torch fallback
(obtained by clearing ``_fused_op`` after construction) on float32 and
bfloat16, and exercises the underlying custom op directly.
"""

import os
from dataclasses import dataclass

import pytest
import torch

from tensorrt_bionemo._torch.custom_ops.adaln_layernorm_sigmoid import get_adaln_layernorm_sigmoid_op
from tensorrt_bionemo._torch.layers.normalization import AdaLN
from tensorrt_bionemo.utils import str_dtype_to_torch


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    torch_dtype: str = "float32"
    seq_len: int = 128
    batch_size: int = 1


def _torch_adaln_layernorm_sigmoid(x, s_scale, s_bias, eps=1e-5):
    """Inline torch reference for ``sigmoid(s_scale) * LN(x, eps) + s_bias``."""
    N = x.shape[-1]
    normed = torch.nn.functional.layer_norm(x, (N,), eps=eps)
    return torch.sigmoid(s_scale) * normed + s_bias


def _init_adaln_weights(m: AdaLN) -> None:
    """Initialize the custom Linear inside AdaLN.

    ``Linear.create_weights`` uses ``torch.empty`` and expects callers to load
    real weights via ``load_weights``. If we skip that, the weights are
    uninitialized garbage and can contain NaN/Inf left over from prior CUDA
    allocations — which silently corrupts test results when forward is run.
    """
    with torch.no_grad():
        torch.nn.init.normal_(m.fused_s_scale_s_bias.weight, std=0.02)
        if m.fused_s_scale_s_bias.bias is not None:
            torch.nn.init.zeros_(m.fused_s_scale_s_bias.bias)
        # s_norm.weight is initialized by nn.LayerNorm.reset_parameters() to 1.


def _copy_weights(src: AdaLN, dst: AdaLN) -> None:
    """Mirror weights from ``src`` into ``dst`` (both AdaLN, same shapes)."""
    src_sd = src.state_dict()
    dst.load_state_dict(src_sd, strict=True)


def _max_rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (actual.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return (diff / denom).max().item()


# ---------------------------------------------------------------------------
# AdaLN module-level: CuteDSL backend vs torch backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dim=768, dim_single_cond=768, torch_dtype="float32"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16"),
        Scenario(dim=384, dim_single_cond=384, torch_dtype="float32"),
        Scenario(dim=384, dim_single_cond=384, torch_dtype="bfloat16"),
        Scenario(dim=512, dim_single_cond=512, torch_dtype="float32"),
        Scenario(dim=512, dim_single_cond=512, torch_dtype="bfloat16"),
        Scenario(dim=1024, dim_single_cond=1024, torch_dtype="float32"),
        Scenario(dim=1024, dim_single_cond=1024, torch_dtype="bfloat16"),
        Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16", batch_size=4),
    ],
)
def test_adaln_cutedsl_matches_torch(sc: Scenario):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    adaln_ref = AdaLN(dim=sc.dim, dim_single_cond=sc.dim_single_cond, dtype=dtype).to(device)
    # Force the inline torch path on the reference module.
    adaln_ref._fused_op = None
    _init_adaln_weights(adaln_ref)
    adaln_cute = AdaLN(dim=sc.dim, dim_single_cond=sc.dim_single_cond, dtype=dtype).to(device)
    _copy_weights(adaln_ref, adaln_cute)

    a = torch.randn(sc.batch_size, sc.seq_len, sc.dim, dtype=dtype, device=device)
    s = torch.randn(sc.batch_size, sc.seq_len, sc.dim_single_cond, dtype=dtype, device=device)

    # The CuteDSL backend is in-place on ``a``. Run each backend on its own
    # copy so the snapshots stay independent.
    a_ref = a.clone()
    a_cute = a.clone()
    a_fp32 = a.float()

    with torch.inference_mode():
        out_ref = adaln_ref.forward(a_ref, s)
        out_cute = adaln_cute.forward(a_cute, s)

    assert out_ref.shape == out_cute.shape
    assert out_ref.dtype == out_cute.dtype

    if dtype == torch.float32:
        torch.testing.assert_close(out_cute, out_ref, atol=1e-4, rtol=1e-4)
    else:
        # bf16: compare both backends against an fp32 ground truth.
        adaln_fp32 = AdaLN(dim=sc.dim, dim_single_cond=sc.dim_single_cond, dtype=torch.float32).to(device)
        adaln_fp32.load_state_dict({k: v.float() for k, v in adaln_ref.state_dict().items()}, strict=True)
        with torch.inference_mode():
            out_fp32 = adaln_fp32.forward(a_fp32, s.float())

        err_ref = (out_ref.float() - out_fp32).abs().max().item()
        err_cute = (out_cute.float() - out_fp32).abs().max().item()
        # Allow 2x the torch backend's quantization error.
        assert err_cute < max(err_ref * 2.0, 0.05), f"CuTe err {err_cute:.3e} too large vs torch err {err_ref:.3e}"


# ---------------------------------------------------------------------------
# Custom op: CuteDSL kernel vs vanilla python reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 19, 1024),
        (1, 64, 768),
        (4, 32, 384),
        (3, 67, 128),
    ],
)
def test_custom_op_matches_vanilla(dtype: torch.dtype, shape: tuple):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, S, D = shape

    x = torch.randn(B, S, D, dtype=dtype, device=device)
    s_scale = torch.randn(B, S, D, dtype=dtype, device=device)
    s_bias = torch.randn(B, S, D, dtype=dtype, device=device)
    eps = 1e-5

    # The fused op is in-place: it mutates ``x``. Snapshot the input first
    # so the reference computation sees the original values.
    x_orig = x.clone()

    fused_op = get_adaln_layernorm_sigmoid_op(dtype)
    out_fused = fused_op(x, s_scale, s_bias, eps=eps)
    out_ref = _torch_adaln_layernorm_sigmoid(x_orig, s_scale, s_bias, eps=eps)

    assert out_fused.shape == out_ref.shape == x_orig.shape
    assert out_fused.dtype == out_ref.dtype == dtype

    out_ref_fp32 = _torch_adaln_layernorm_sigmoid(x_orig.float(), s_scale.float(), s_bias.float(), eps=eps)

    diff_fused = (out_fused.float() - out_ref_fp32).abs().max().item()
    diff_torch = (out_ref.float() - out_ref_fp32).abs().max().item()

    if dtype == torch.float32:
        torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)
    else:
        # Allow 2x the torch backend's quantization error.
        assert diff_fused < max(diff_torch * 2.0, 0.05), (
            f"fused err {diff_fused:.3e} too large vs torch err {diff_torch:.3e}"
        )


# ---------------------------------------------------------------------------
# Forcing the torch fallback path must produce a bit-exact reference run
# ---------------------------------------------------------------------------


def test_adaln_torch_backend_explicit():
    """Clearing ``_fused_op`` forces the inline torch path; two such modules
    with identical weights must agree bit-for-bit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")

    a_dim, s_dim = 768, 768
    adaln_a = AdaLN(dim=a_dim, dim_single_cond=s_dim).to(device)
    adaln_a._fused_op = None
    _init_adaln_weights(adaln_a)
    adaln_b = AdaLN(dim=a_dim, dim_single_cond=s_dim).to(device)
    adaln_b._fused_op = None
    _copy_weights(adaln_a, adaln_b)

    a = torch.randn(1, 64, a_dim, device=device)
    s = torch.randn(1, 64, s_dim, device=device)

    with torch.inference_mode():
        out_a = adaln_a.forward(a, s)
        out_b = adaln_b.forward(a, s)

    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Default backend is CuteDSL with silent torch fallback
# ---------------------------------------------------------------------------


def test_adaln_default_backend_is_cutedsl():
    """No explicit ``backend`` ⇒ kernel-backed module (or transparent
    fallback if the kernel can't load on this SM/dtype). Either way the
    output must match a forced-torch reference within kernel tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dim = 768

    adaln_default = AdaLN(dim=dim, dim_single_cond=dim).to(device)  # CuteDSL
    adaln_torch = AdaLN(dim=dim, dim_single_cond=dim).to(device)
    # Force the inline torch path on the reference module.
    adaln_torch._fused_op = None
    _init_adaln_weights(adaln_torch)
    _copy_weights(adaln_torch, adaln_default)

    a = torch.randn(1, 64, dim, device=device)
    s = torch.randn(1, 64, dim, device=device)

    # CuteDSL backend mutates ``a`` in place — give each forward its own copy.
    a_default = a.clone()
    a_torch = a.clone()

    with torch.inference_mode():
        out_default = adaln_default.forward(a_default, s)
        out_torch = adaln_torch.forward(a_torch, s)

    # fp32 default dtype — CuTe kernel matches torch closely.
    torch.testing.assert_close(out_default, out_torch, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Multiplicity broadcast — s_scale / s_bias have size 1 on one leading dim
# (e.g. x is [B, S, I, D] and s is [B, 1, I, D]).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "x_shape, s_shape",
    [
        # Multi-sample case: multiplicity dim is second from front.
        ((1, 5, 128, 768), (1, 1, 128, 768)),
        ((2, 8, 128, 384), (2, 1, 128, 384)),
        ((1, 10, 256, 768), (1, 1, 256, 768)),
        ((1, 4, 512, 768), (1, 1, 512, 768)),
        # Long seq, large multiplicity.
        ((1, 32, 128, 256), (1, 1, 128, 256)),
        # Broadcast on the batch dim.
        ((4, 128, 768), (1, 128, 768)),
        ((4, 5, 64, 256), (1, 5, 64, 256)),
        # 5-D layout ``(B, S, I, J, D)`` with multiplicity at dim 1.
        ((1, 5, 28, 32, 128), (1, 1, 28, 32, 128)),
        # 5-D layout but broadcast on an interior dim (dim 2).
        ((2, 4, 7, 128, 384), (2, 4, 1, 128, 384)),
        ((2, 6, 8, 384), (2, 1, 8, 384)),
        # Small / odd ``inner = prod(x.shape[bcast_dim + 1:-1])``.
        ((1, 4, 3, 768), (1, 1, 3, 768)),  # inner=3
        ((1, 5, 7, 384), (1, 1, 7, 384)),  # inner=7
        ((2, 4, 5, 128), (2, 1, 5, 128)),  # inner=5
        ((1, 8, 1, 768), (1, 1, 1, 768)),  # inner=1
        ((1, 3, 6, 384), (1, 1, 6, 384)),  # inner=6
    ],
)
def test_custom_op_broadcast(dtype: torch.dtype, x_shape: tuple, s_shape: tuple):
    """Kernel must produce torch-equivalent output when s_scale / s_bias
    broadcast against one leading dim of x."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")

    x = torch.randn(*x_shape, dtype=dtype, device=device)
    s_scale = torch.randn(*s_shape, dtype=dtype, device=device)
    s_bias = torch.randn(*s_shape, dtype=dtype, device=device)
    eps = 1e-5

    # Run on a clone since the kernel may write in-place when out is None.
    x_orig = x.clone()
    out = torch.empty_like(x)

    fused_op = get_adaln_layernorm_sigmoid_op(dtype)
    out_fused = fused_op(x_orig.clone(), s_scale, s_bias, out=out, eps=eps)

    # Reference: torch handles size-1 broadcast natively.
    out_ref = _torch_adaln_layernorm_sigmoid(x_orig, s_scale, s_bias, eps=eps)

    assert out_fused.shape == out_ref.shape == x_orig.shape
    assert out_fused.dtype == dtype

    out_ref_fp32 = _torch_adaln_layernorm_sigmoid(x_orig.float(), s_scale.float(), s_bias.float(), eps=eps)

    if dtype == torch.float32:
        torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)
    else:
        diff_fused = (out_fused.float() - out_ref_fp32).abs().max().item()
        diff_torch = (out_ref.float() - out_ref_fp32).abs().max().item()
        assert diff_fused < max(diff_torch * 2.0, 0.05), (
            f"fused err {diff_fused:.3e} too large vs torch err {diff_torch:.3e}"
        )


def test_adaln_multisample_does_not_fall_back():
    """Regression: with x=[B,S,I,D] and s=[B,1,I,D], the CuTeDSL kernel must
    actually run rather than get disabled inside ``AdaLN.forward``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dim, dim_cond = 768, 768
    B, S, I = 1, 5, 128

    module = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=torch.bfloat16).to(device)
    _init_adaln_weights(module)
    assert module._fused_op is not None, "kernel must initialize on supported SM"

    a = torch.randn(B, S, I, dim, dtype=torch.bfloat16, device=device)
    s = torch.randn(B, 1, I, dim_cond, dtype=torch.bfloat16, device=device)

    with torch.inference_mode():
        out = module(a.clone(), s)

    assert module._fused_op is not None, (
        "AdaLN.forward disabled the kernel mid-forward — multi-sample "
        "broadcast regression (kernel should handle s_scale/s_bias with "
        "size-1 multiplicity dim)"
    )
    assert out.shape == a.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_samples", [5, 10])
def test_adaln_multisample_matches_torch(dtype: torch.dtype, num_samples: int):
    """End-to-end: AdaLN with multi-sample input should give same numerics
    via CuteDSL backend as via torch backend."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    device = torch.device("cuda")
    dim, dim_cond = 768, 768
    B, I = 1, 128

    adaln_ref = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=dtype).to(device)
    # Force the inline torch path on the reference module.
    adaln_ref._fused_op = None
    _init_adaln_weights(adaln_ref)
    adaln_cute = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=dtype).to(device)
    _copy_weights(adaln_ref, adaln_cute)

    a = torch.randn(B, num_samples, I, dim, dtype=dtype, device=device)
    s = torch.randn(B, 1, I, dim_cond, dtype=dtype, device=device)

    a_ref = a.clone()
    a_cute = a.clone()

    with torch.inference_mode():
        out_ref = adaln_ref.forward(a_ref, s)
        out_cute = adaln_cute.forward(a_cute, s)

    assert adaln_cute._fused_op is not None, "kernel disabled — broadcast fallback regression"
    assert out_ref.shape == out_cute.shape == a.shape

    if dtype == torch.float32:
        torch.testing.assert_close(out_cute, out_ref, atol=1e-4, rtol=1e-4)
    else:
        adaln_fp32 = AdaLN(dim=dim, dim_single_cond=dim_cond, dtype=torch.float32).to(device)
        adaln_fp32.load_state_dict({k: v.float() for k, v in adaln_ref.state_dict().items()}, strict=True)
        with torch.inference_mode():
            out_fp32 = adaln_fp32.forward(a.float(), s.float())
        err_ref = (out_ref.float() - out_fp32).abs().max().item()
        err_cute = (out_cute.float() - out_fp32).abs().max().item()
        assert err_cute < max(err_ref * 2.0, 0.05), f"CuTe err {err_cute:.3e} vs torch err {err_ref:.3e}"
