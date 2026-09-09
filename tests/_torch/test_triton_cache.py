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

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from bionemo_ir.dsl_kernels.cache_base import make_driver_launcher
from bionemo_ir.dsl_kernels.triton.fused_ln_proj_moveaxis_pad import (
    FusedLNProjMoveaxisPad,
    _fused_ln_proj_moveaxis_pad_kernel,
)
from bionemo_ir.dsl_kernels.triton.fused_swiglu import FusedSwiGLU, _fused_swiglu_kernel
from bionemo_ir.dsl_kernels.triton.moveaxis_pad import MoveaxisPad, _moveaxis_pad_kernel
from bionemo_ir.dsl_kernels.triton_cache import (
    _DRIVER_TRITON_OK,
    _SUPPORTED_TRITON_VERSIONS,
    TritonKernelCache,
    _looks_like_compiled_kernel,
    value_specialized_params,
)

# Kernels whose one CUBIN, compiled from dummy arguments, is reused for every shape.
DRIVER_LAUNCHED_KERNELS = [
    _fused_ln_proj_moveaxis_pad_kernel,
    _moveaxis_pad_kernel,
    _fused_swiglu_kernel,
]

# Skip rather than assert: an unvalidated Triton is a property of the image, not a
# defect, and shape-agnostic launch is covered on any version by
# ``test_cached_cubin_is_correct_for_shapes_it_was_not_compiled_with``.
requires_driver_path = pytest.mark.skipif(
    not _DRIVER_TRITON_OK,
    reason=(
        f"direct CUBIN launcher needs Triton "
        f"{' / '.join('.'.join(map(str, v)) for v in _SUPPORTED_TRITON_VERSIONS)}, "
        f"found {triton.__version__}"
    ),
)


def test_driver_abi_versions_stay_within_the_declared_triton_range() -> None:
    """Every ABI the fast path claims must be a Triton the package can install.

    These drifted once: requirements widened to 3.7 while the launcher still gated on
    3.6, silently costing a supported install its fast path.
    """
    requirements = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text()
    (pin,) = [line for line in requirements.splitlines() if line.startswith("triton")]
    allowed = SpecifierSet(pin.removeprefix("triton"))

    assert _SUPPORTED_TRITON_VERSIONS, "the fast path must claim at least one ABI"
    for major, minor in _SUPPORTED_TRITON_VERSIONS:
        assert allowed.contains(Version(f"{major}.{minor}.0")), (
            f"driver path claims Triton {major}.{minor}, which {pin!r} does not allow"
        )


@pytest.mark.parametrize("jit_fn", DRIVER_LAUNCHED_KERNELS, ids=lambda fn: fn.fn.__name__)
def test_driver_launched_kernels_declare_do_not_specialize(jit_fn) -> None:
    """A reused CUBIN must not be specialized on its compile-time argument values.

    Triton derives ``divisible_by_16`` from the dummy arguments, which corrupts output
    for runtime values that are not multiples of 16, and ``equal_to_1``, which drops
    the argument and shifts every later ``params[i]`` assignment.
    """
    unprotected = [
        param.name
        for param in jit_fn.params
        if not param.is_constexpr and not param.do_not_specialize and not param.name.endswith("_ptr")
    ]
    assert unprotected == [], (
        f"{jit_fn.fn.__name__} is launched from a cached CUBIN, so scalar parameter(s) "
        f"{unprotected} must be listed in do_not_specialize"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_value_specialized_params_flags_unprotected_scalars() -> None:
    """The guard that keeps an unsafe kernel off the driver path."""
    op = FusedLNProjMoveaxisPad(D=128, H=16, dtype=torch.bfloat16)
    dummy = op._make_dummy_args(torch.bfloat16)

    assert value_specialized_params(_fused_ln_proj_moveaxis_pad_kernel, dummy) == []

    # A scalar in a pointer slot is not exempt; unreadable parameters are unverifiable.
    scalar_for_pointer = (dummy[0], 1, *dummy[2:])
    assert "w_ln_ptr" in value_specialized_params(_fused_ln_proj_moveaxis_pad_kernel, scalar_for_pointer)

    class _NoParams:
        pass

    assert value_specialized_params(_NoParams(), dummy) is None


@triton.jit
def _add_n_kernel(dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, offs.to(tl.float32) + n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_value_specialized_compile_does_not_wrap_dummy_cubin() -> None:
    """An unprotected scalar must not be launched from the dummy CompiledKernel.

    Compiling with n=1 lets Triton fold equal_to_1 into the CUBIN. CachedKernel.launch()
    used to call .run() on that CUBIN, so n=5 still added 1. The JIT path recompiles.
    """
    block = 16
    dummy_out = torch.empty(block, device="cuda", dtype=torch.float32)
    dummy = (dummy_out, 1)
    assert value_specialized_params(_add_n_kernel, dummy) == ["n"]

    kernel = TritonKernelCache().compile(_add_n_kernel, dummy, (1,), {"BLOCK": block})
    assert kernel.driver is None
    assert kernel.compiled is _add_n_kernel
    assert not _looks_like_compiled_kernel(kernel.compiled)

    out = torch.empty(block, device="cuda", dtype=torch.float32)
    kernel.launch((1,), out, 5, block)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.arange(block, device="cuda", dtype=torch.float32) + 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unreadable_params_do_not_wrap_dummy_cubin(monkeypatch: pytest.MonkeyPatch) -> None:
    """When specialization cannot be inspected, launch must not reuse the dummy CUBIN."""
    monkeypatch.setattr(
        "bionemo_ir.dsl_kernels.triton_cache.value_specialized_params",
        lambda jit_fn, dummy_args: None,
    )
    block = 16
    dummy = (torch.empty(block, device="cuda", dtype=torch.float32), 1)
    kernel = TritonKernelCache().compile(_add_n_kernel, dummy, (1,), {"BLOCK": block})
    assert kernel.driver is None
    assert kernel.compiled is _add_n_kernel

    out = torch.empty(block, device="cuda", dtype=torch.float32)
    kernel.launch((1,), out, 5, block)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.arange(block, device="cuda", dtype=torch.float32) + 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@requires_driver_path
def test_driver_fast_path_launches_fused_swiglu() -> None:
    pytest.importorskip("cuda.bindings.driver")

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
@requires_driver_path
def test_fused_ln_projection_driver_is_deterministic_in_cuda_graph() -> None:
    pytest.importorskip("cuda.bindings.driver")

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
    # Graph and eager run the same kernel, so agreeing shows capture safety, not
    # correctness: at j=199 this passed while the kernel corrupted that very shape.
    expected = _ln_proj_moveaxis_pad_reference(z, ln_weight, ln_bias, proj_weight, graph_output.shape[-1])
    torch.testing.assert_close(graph_output.float(), expected, rtol=0.05, atol=0.05)


def _ln_proj_moveaxis_pad_reference(
    z: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    j_padded: int,
) -> torch.Tensor:
    out = F.layer_norm(z.float(), [z.shape[-1]], ln_weight.float(), ln_bias.float(), eps=1e-5)
    out = F.linear(out, proj_weight.float()).movedim(-1, -3)
    return F.pad(out, (0, j_padded - out.shape[-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dim_d", "heads", "rms_norm", "norm_dtype"),
    [
        pytest.param(196, 4, False, torch.float32, id="layernorm-d196-mixed"),
        pytest.param(384, 12, True, torch.bfloat16, id="rmsnorm-d384-velocity"),
    ],
)
def test_fused_ln_projection_supports_non_power_of_two_feature_dims(
    dim_d: int,
    heads: int,
    rms_norm: bool,
    norm_dtype: torch.dtype,
) -> None:
    """Padded reduction lanes and the final partial K tile must not affect output."""
    tokens = 19
    torch.manual_seed(123)
    z = torch.randn(1, tokens, tokens, dim_d, device="cuda", dtype=torch.bfloat16)
    ln_weight = torch.randn(dim_d, device="cuda", dtype=norm_dtype)
    ln_bias = None if rms_norm else torch.randn(dim_d, device="cuda", dtype=norm_dtype)
    proj_weight = torch.randn(heads, dim_d, device="cuda", dtype=z.dtype) * 0.05

    op = FusedLNProjMoveaxisPad(D=dim_d, H=heads, dtype=z.dtype, rms_norm=rms_norm)
    actual = op(z, ln_weight, ln_bias, proj_weight, multiple=8)
    torch.cuda.synchronize()

    if rms_norm:
        normalized = F.rms_norm(z.float(), [dim_d], ln_weight.float(), eps=1e-5)
    else:
        assert ln_bias is not None
        normalized = F.layer_norm(z.float(), [dim_d], ln_weight.float(), ln_bias.float(), eps=1e-5)
    expected = F.linear(normalized, proj_weight.float()).movedim(-1, -3)
    expected = F.pad(expected, (0, actual.shape[-1] - tokens))

    assert torch.count_nonzero(actual[..., tokens:]) == 0
    torch.testing.assert_close(actual.float(), expected, rtol=0.05, atol=0.05)


# 30 and 42 are the Boltz-2 token counts that crashed, neither a multiple of 16.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [30, 42, 199, 256])
def test_cached_cubin_is_correct_for_shapes_it_was_not_compiled_with(num_tokens: int) -> None:
    """Regression: launching the cached CUBIN must not depend on the dummy shapes.

    Compiles from ``_make_dummy_args``, where every scalar is a multiple of 16, then
    launches that CUBIN with real token counts as ``DriverLauncher`` does. Before
    ``do_not_specialize`` this produced wrong values and NaNs from out-of-bounds reads.
    """
    pytest.importorskip("cuda.bindings.driver")
    dim_d, heads, tile_j, heads_per_blk = 128, 16, 32, 16
    constexprs = {
        "TILE_J": tile_j,
        "TILE_K": 128,
        "BLOCK_D": dim_d,
        "DIM_D": dim_d,
        "NUM_HEADS": heads,
        "HEADS_PER_BLK": heads_per_blk,
        "EPS": 1e-5,
        "ELEMENTWISE_AFFINE": True,
        "RMS_NORM": False,
    }
    torch.manual_seed(0)
    ln_weight = torch.randn(dim_d, device="cuda", dtype=torch.bfloat16)
    ln_bias = torch.randn(dim_d, device="cuda", dtype=torch.bfloat16)
    proj_weight = torch.randn(heads, dim_d, device="cuda", dtype=torch.bfloat16) * 0.05

    op = FusedLNProjMoveaxisPad(D=dim_d, H=heads, dtype=torch.bfloat16)
    dummy_z = torch.empty(1, 2, tile_j, dim_d, device="cuda", dtype=torch.bfloat16)
    dummy_out = torch.empty(1, heads, 2, tile_j, device="cuda", dtype=torch.bfloat16)
    compiled = _fused_ln_proj_moveaxis_pad_kernel[(1, 2, 1)](
        dummy_z,
        ln_weight,
        ln_bias,
        proj_weight,
        dummy_out,
        *op._make_dummy_args(torch.bfloat16)[5:],
        **constexprs,
    )
    torch.cuda.synchronize()
    launcher = make_driver_launcher(compiled)
    assert launcher is not None

    j_padded = ((num_tokens + 7) // 8) * 8
    z = torch.randn(1, num_tokens, num_tokens, dim_d, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, heads, num_tokens, j_padded, device="cuda", dtype=torch.bfloat16)
    values = [
        z.data_ptr(),
        ln_weight.data_ptr(),
        ln_bias.data_ptr(),
        proj_weight.data_ptr(),
        out.data_ptr(),
        num_tokens,
        j_padded,
        *z.stride()[:3],
        *out.stride()[:3],
    ]
    # Triton appends scratch pointers after the declared arguments; they stay NULL.
    for param, value in zip(launcher.params, values, strict=False):
        param.value = value
    launcher.launch(triton.cdiv(j_padded, tile_j), num_tokens, triton.cdiv(heads, heads_per_blk))
    torch.cuda.synchronize()

    expected = _ln_proj_moveaxis_pad_reference(z, ln_weight, ln_bias, proj_weight, j_padded)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.float(), expected, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("heads", [3, 4, 6, 13, 16])
def test_moveaxis_pad_masks_the_head_axis(heads: int) -> None:
    """Regression: BLOCK_H rounds H up to a power of two, so h needs a mask.

    Without one the kernel stores ``(BLOCK_H - H) * I * J_padded`` elements past the
    output, which a canary allocated right behind it catches.
    """
    tokens_i, tokens_j = 8, 30
    j_padded = ((tokens_j + 7) // 8) * 8
    n_out = heads * tokens_i * j_padded

    op = MoveaxisPad(H=heads, dtype=torch.bfloat16)
    torch.manual_seed(0)
    x = torch.randn(1, tokens_i, tokens_j, heads, device="cuda", dtype=torch.bfloat16)

    arena = torch.full((n_out * 4,), 7.0, device="cuda", dtype=torch.bfloat16)
    canary_before = arena[n_out:].clone()

    real_empty = torch.empty

    def fake_empty(*size, **kwargs):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = tuple(size[0])
        if size == (1, heads, tokens_i, j_padded):
            return arena[:n_out].view(1, heads, tokens_i, j_padded)
        return real_empty(*size, **kwargs)

    torch.empty = fake_empty
    try:
        actual = op(x, multiple=8)
    finally:
        torch.empty = real_empty
    torch.cuda.synchronize()

    assert torch.equal(arena[n_out:], canary_before), "kernel wrote past the end of the output"
    expected = F.pad(x.movedim(-1, -3), (0, j_padded - tokens_j))
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moveaxis_pad_rejects_more_heads_than_compiled_for() -> None:
    op = MoveaxisPad(H=4, dtype=torch.bfloat16)
    too_many = torch.randn(1, 8, 30, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="heads"):
        op(too_many, multiple=8)
