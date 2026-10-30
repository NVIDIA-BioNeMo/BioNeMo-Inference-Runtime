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
"""The transition with its LayerNorm in one SM90 kernel: the kernel's valid-row tiling, flat rows, the runtime mask
and residual at C = 384, H = 768 (ReLU) and every other ReLU shape, the weight pack, and the op from source and
CUBINs, against an fp32 reference. tests/_torch/test_transition_w_norm.py runs the transition layers on it."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.custom_ops import transition_mlp_w_norm as w_norm
from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import cutedsl as w_norm_cutedsl
from bionemo_ir._torch.custom_ops.transition_mlp_w_norm._cubin import TransitionMlpWNormCubinExecutable
from bionemo_ir._torch.custom_ops.transition_mlp_w_norm.cutedsl import TransitionMlpWNormCuTe
from bionemo_ir._torch.custom_ops.trimul_kf_k3.ops import centre_rows_bf16
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelVariantUnavailable
from bionemo_ir._torch.utils.kernel import _cutedsl_kernel_library as library_runtime
from tests._torch import (
    cutedsl_test_modes,
    require_cubin_family,
    run_cutedsl_test_mode,
    skip_if_not_sm90,
    source_module_available,
)

_SOURCE = "bionemo_ir._torch.custom_ops.transition_mlp_w_norm._source"
_MODES = cutedsl_test_modes(_SOURCE)
_requires_source = pytest.mark.skipif(not source_module_available(_SOURCE), reason="launches the kernel from source")
C, H, EPS = 384, 768, 1e-5
ATOL, RTOL = 5e-2, 2e-2  # tests/ops/test_transition_mlp.py's tolerance
_EXE = {}


def _compiled():
    """The kernel compiled from source, launched directly with explicit per-item lengths."""
    return _compiled_shape(C, H)


def _full_tile_wgs(width, hidden, activation="relu"):
    """The persistent launch's warpgroups per row tile: all of the kernel's."""
    from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source

    return _source.launch_geometry(width, hidden, activation)["tile_m"] // 64


def _params(seed=0):
    """gamma, beta, w1, b1, w2, b2 at the released weights' scale."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=1.0, offset=0.0):
        return (offset + scale * torch.randn(*shape, device="cuda", generator=g)).to(torch.bfloat16)

    return (
        r(C, scale=0.1, offset=1.0),  # gamma
        r(C, scale=0.1),  # beta
        r(H, C, scale=C**-0.5),  # w1
        r(H, scale=0.1),  # b1
        r(C, H, scale=H**-0.5),  # w2
        r(C, scale=0.1),  # b2
    )


def _reference(z, mask, gamma, beta, w1, b1, w2, b2, residual=True):
    """fp32 ``z + mask * u`` with the residual, ``mask * u`` without it; ``mask`` ``None`` keeps every row."""
    ones = torch.ones(z.shape[:-1], dtype=torch.float32, device=z.device)
    out = w_norm.transition_mlp_w_norm_reference(z, ones if mask is None else mask, gamma, beta, w1, b1, w2, b2, EPS)
    return out if residual else out - z.float()


def _run(z, mask, params, out, n_valid=None, *, flat=False, residual=True, layout=None):
    """Pair maps: every row (``n_valid`` None), or in place over the rows with i, j < n_valid[b] (``out`` is ``z``).
    ``flat``: every row of ``z`` ``[..., C]`` as flat rows. ``mask`` ``None`` keeps every row. ``layout``:
    ``(tile_wgs, split)``, by default the persistent launch."""
    from bionemo_ir._torch.utils.kernel import launch_compiled_kernel

    pack = w_norm.pack_weights(*params)
    rows = z.numel() // C
    if flat:
        n_valid_address, batch, tokens = 0, 1, 1
    else:
        batch, tokens = z.shape[0], z.shape[1]
        if n_valid is None:
            n_valid = torch.full((batch,), tokens, dtype=torch.int32, device=z.device)
        n_valid_address = n_valid.data_ptr()
    mask_address = 0 if mask is None else mask.data_ptr()
    launch_compiled_kernel(
        _compiled(),
        z.view(-1),
        mask_address,
        pack.image,
        pack.vec,
        out.view(-1),
        n_valid_address,
        rows,
        tokens,
        batch,
        EPS,
        int(residual),
        *_launch_layout(C, H, "relu", layout),
    )
    return out


def _launch_layout(width, hidden, activation, layout):
    """The kernel's trailing ``split, tile_wgs`` for ``layout`` ``(tile_wgs, split)`` or the persistent launch."""
    tile_wgs, split = layout or (_full_tile_wgs(width, hidden, activation), 1)
    return split, tile_wgs


def _problem(batch, tokens, lengths, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + 1000 * batch + tokens)
    z = torch.randn(batch, tokens, tokens, C, device="cuda", generator=g).to(torch.bfloat16)
    mask = torch.ones(batch, tokens, tokens, 1, device="cuda", dtype=torch.bfloat16)
    for b, length in enumerate(lengths or []):
        valid = torch.arange(tokens, device="cuda") < length
        mask[b, ..., 0] = (valid[:, None] & valid[None, :]).to(torch.bfloat16)
    return z, mask


def _op(width: int = C, hidden: int = H, activation: str = "relu") -> w_norm.TransitionMlpWNormOp:
    """The op for a shipped or candidate shape."""
    op = w_norm.get_transition_mlp_w_norm_op(torch.bfloat16, width, hidden, activation, candidates=True)
    assert op is not None, (width, hidden, activation)
    return op


def _run_op(mode, monkeypatch, z, mask, params, out=None):
    """The ReLU op at C = 384 over every row, through exactly one of the source and CUBIN paths."""
    if mode == "cubin":
        require_cubin_family("transition_mlp_w_norm")
    op = _op()
    pack = w_norm.pack_weights(*params)
    return run_cutedsl_test_mode(
        mode, monkeypatch, TransitionMlpWNormCuTe, w_norm_cutedsl, lambda: op(z, mask, pack, EPS, out=out)
    )


# (batch, tokens, per-item valid lengths): partial last tiles (25 rows leave the second warpgroup none), fewer tiles
# than SMs, more tiles than SMs, masked rows inside a tile and whole masked items.
_CASES = [
    (1, 5, None),
    (1, 11, [7]),
    (2, 33, [33, 20]),
    (1, 104, [100]),
    (3, 168, [168, 0, 50]),
    (1, 200, None),
    (2, 256, [256, 129]),
]


# ---- the kernel, launched directly with per-item lengths ------------------------------------------------------------


@_requires_source
@pytest.mark.parametrize(("batch", "tokens", "lengths"), _CASES)
def test_matches_fp32_reference_and_passes_masked_rows_through(batch, tokens, lengths):
    skip_if_not_sm90()
    params = _params()
    z, mask = _problem(batch, tokens, lengths)
    out = _run(z, mask, params, torch.full_like(z, float("nan")))
    torch.testing.assert_close(out.float(), _reference(z, mask, *params), atol=ATOL, rtol=RTOL)
    masked = mask[..., 0] == 0
    assert torch.equal(out[masked], z[masked]), "a row whose mask is 0 must come out equal to z"


@_requires_source
@pytest.mark.parametrize(("batch", "tokens", "lengths"), [(1, 11, [5]), (3, 168, [168, 0, 50]), (2, 200, [9, 200])])
def test_valid_rows_in_place(batch, tokens, lengths):
    """Only the rows with i, j < n_valid[b] are computed; the others are neither read nor written."""
    skip_if_not_sm90()
    params = _params()
    z, mask = _problem(batch, tokens, lengths)
    ref = _reference(z, mask, *params)
    zc = z.clone()
    _run(zc, mask, params, zc, torch.tensor(lengths, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(zc.float(), ref, atol=ATOL, rtol=RTOL)
    masked = mask[..., 0] == 0
    assert torch.equal(zc[masked], z[masked]), "rows outside n_valid must be left as they are"


@_requires_source
def test_mask_scales_the_update_only():
    skip_if_not_sm90()
    params = _params(1)
    z, mask = _problem(1, 64, None, seed=1)
    g = torch.Generator(device="cuda").manual_seed(7)
    picks = torch.rand(mask.shape, device="cuda", generator=g)
    mask = torch.where(picks < 0.1, 0.5, torch.where(picks < 0.2, 0.0, 1.0)).to(torch.bfloat16)
    out = _run(z, mask, params, torch.empty_like(z))
    torch.testing.assert_close(out.float(), _reference(z, mask, *params), atol=ATOL, rtol=RTOL)
    masked = mask[..., 0] == 0
    assert torch.equal(out[masked], z[masked])


def _large_mean_atol(z: torch.Tensor) -> float:
    """The bf16 output's own rounding grows with |z|: half a unit in the last place of the largest |z|."""
    return float(ATOL + 0.5 * 2.0 ** (torch.floor(torch.log2(z.float().abs().max())) - 7))


@_requires_source
@pytest.mark.parametrize(("mean", "std"), [(20.0, 1.0), (-50.0, 2.0), (100.0, 1.0)])
def test_rows_with_a_large_mean(mean, std):
    """|mean| >= 10 std: GEMM 1 multiplies z itself, and the centred bf16 rows of W1 diag(gamma) keep z's mean out."""
    skip_if_not_sm90()
    params = _params(3)
    g = torch.Generator(device="cuda").manual_seed(3)
    z = (mean + std * torch.randn(1, 64, 64, C, device="cuda", generator=g)).to(torch.bfloat16)
    mask = torch.ones(1, 64, 64, 1, device="cuda", dtype=torch.bfloat16)
    ref = _reference(z, mask, *params)
    out = _run(z, mask, params, torch.empty_like(z))
    torch.testing.assert_close(out.float(), ref, atol=_large_mean_atol(z), rtol=RTOL)


def _random_mask(shape, seed):
    """Mostly 1, with 0.5 and 0 rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    picks = torch.rand(shape, device="cuda", generator=g)
    return torch.where(picks < 0.1, 0.5, torch.where(picks < 0.2, 0.0, 1.0)).to(torch.bfloat16)


@_requires_source
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("shape", [(1, 1, 37), (2, 16, 75), (1, 128, 133)])
def test_flat_rows_with_and_without_mask_and_residual(shape, with_mask, residual):
    """Flat rows ``[B, S, N, C]``: every row is computed; a 0 mask gives z with the residual and 0 without it."""
    skip_if_not_sm90()
    params = _params(8)
    g = torch.Generator(device="cuda").manual_seed(8)
    z = torch.randn(*shape, C, device="cuda", generator=g).to(torch.bfloat16)
    mask = _random_mask(shape, 9) if with_mask else None
    out = _run(z, mask, params, torch.full_like(z, float("nan")), flat=True, residual=residual)
    torch.testing.assert_close(out.float(), _reference(z, mask, *params, residual=residual), atol=ATOL, rtol=RTOL)
    if with_mask:
        masked = mask == 0
        assert torch.equal(out[masked], z[masked] if residual else torch.zeros_like(z[masked]))


@_requires_source
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize(("batch", "tokens", "lengths"), [(1, 11, [5]), (3, 168, [168, 0, 50])])
def test_valid_rows_with_and_without_mask_and_residual(batch, tokens, lengths, with_mask, residual):
    """In place over the rows with i, j < n_valid[b]: the others keep z whatever the mask and residual."""
    skip_if_not_sm90()
    params = _params(10)
    z, pair_mask = _problem(batch, tokens, lengths, seed=10)
    mask = pair_mask * _random_mask(pair_mask.shape, 11) if with_mask else None
    zc = z.clone()
    _run(zc, mask, params, zc, torch.tensor(lengths, dtype=torch.int32, device="cuda"), residual=residual)
    inside = pair_mask[..., 0] != 0
    ref = _reference(z, mask, *params, residual=residual)
    torch.testing.assert_close(zc[inside].float(), ref[inside], atol=ATOL, rtol=RTOL)
    assert torch.equal(zc[~inside], z[~inside]), "rows outside n_valid must be left as they are"


@_requires_source
@pytest.mark.parametrize(("batch", "tokens"), [(1, 104), (2, 33), (3, 64)])
def test_flat_rows_match_the_pair_layout_bit_for_bit(batch, tokens):
    """A whole pair map gives the same bits as flat rows and as pair rows with every item full."""
    skip_if_not_sm90()
    params = _params(12)
    z, _ = _problem(batch, tokens, None, seed=12)
    mask = _random_mask((batch, tokens, tokens, 1), 13)
    pair = _run(z, mask, params, torch.empty_like(z))
    flat = _run(z, mask, params, torch.empty_like(z), flat=True)
    assert torch.equal(pair, flat)


@_requires_source
def test_is_deterministic():
    skip_if_not_sm90()
    params = _params(2)
    z, mask = _problem(2, 136, [136, 70], seed=2)
    first = _run(z, mask, params, torch.empty_like(z))
    second = _run(z, mask, params, torch.empty_like(z))
    assert torch.equal(first, second)


# ---- every ReLU shape, launched directly -----------------------------------------------------------------------------

#: (C, H): the in-scope ReLU shapes, then the remaining 2C hidden widths at C = 128 and 256.
_RELU_SHAPES = [(384, 768), (128, 512), (256, 1024), (64, 256), (64, 128), (128, 256), (256, 512)]


def _compiled_shape(width, hidden, persistent=False):
    """The kernel for one ReLU shape compiled from source through the kernel disk cache, which can be seeded outside
    a GPU lock: the image of every layout, or the persistent image (``persistent``)."""
    if (width, hidden, persistent) not in _EXE:
        from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source
        from bionemo_ir.dsl_kernels.cute_cache import CuteKernelCache

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        cache = CuteKernelCache()
        key = ("test_transition_mlp_w_norm_v2", width, hidden, persistent, num_sms)
        executable = cache.load_from_cache(key)
        if executable is None:
            executable = _source.compile_transition_mlp_w_norm_source(
                cache.compile, width, hidden, num_sms, persistent=persistent
            )
            cache.save_to_cache(key, executable)
        _EXE[(width, hidden, persistent)] = executable
    return _EXE[(width, hidden, persistent)]


def _shape_params(width, hidden, seed):
    """gamma, beta, w1, b1, w2, b2 for one shape, at the released weights' scale."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, scale=1.0, offset=0.0):
        return (offset + scale * torch.randn(*shape, device="cuda", generator=g)).to(torch.bfloat16)

    return (
        r(width, scale=0.1, offset=1.0),
        r(width, scale=0.1),
        r(hidden, width, scale=width**-0.5),
        r(hidden, scale=0.1),
        r(width, hidden, scale=hidden**-0.5),
        r(width, scale=0.1),
    )


def _launch(width, hidden, z, mask, params, out, n_valid, residual, layout=None, persistent=False):
    """Flat rows when ``n_valid`` is None, else pair maps ``[B, N, N, width]`` over their valid rows; ``layout`` as
    :func:`_run`'s; ``persistent``: the persistent image (it takes the full tile at split 1 only)."""
    from bionemo_ir._torch.utils.kernel import launch_compiled_kernel

    pack = w_norm.pack_weights(*params)
    batch, tokens = (1, 1) if n_valid is None else (z.shape[0], z.shape[1])
    launch_compiled_kernel(
        _compiled_shape(width, hidden, persistent),
        z.view(-1),
        0 if mask is None else mask.data_ptr(),
        pack.image,
        pack.vec,
        out.view(-1),
        0 if n_valid is None else n_valid.data_ptr(),
        z.numel() // width,
        tokens,
        batch,
        EPS,
        int(residual),
        *_launch_layout(width, hidden, "relu", layout),
    )
    return out


@_requires_source
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize(("width", "hidden"), _RELU_SHAPES)
def test_every_relu_shape_on_pair_maps_and_flat_rows(width, hidden, with_mask, residual):
    """Padded pair maps in place (rows outside n_valid untouched) and flat MSA-like rows, against fp32."""
    skip_if_not_sm90()
    params = _shape_params(width, hidden, seed=width + hidden)
    g = torch.Generator(device="cuda").manual_seed(width)
    # pair maps: a full item, an empty one and a padded one across several tiles
    batch, tokens, lengths = 3, 40, [40, 0, 23]
    z = torch.randn(batch, tokens, tokens, width, device="cuda", generator=g).to(torch.bfloat16)
    inside = torch.zeros(batch, tokens, tokens, dtype=torch.bool, device="cuda")
    for b, length in enumerate(lengths):
        valid = torch.arange(tokens, device="cuda") < length
        inside[b] = valid[:, None] & valid[None, :]
    mask = (_random_mask(inside.shape, width) * inside).to(torch.bfloat16) if with_mask else None
    out = z.clone()
    _launch(width, hidden, out, mask, params, out, torch.tensor(lengths, dtype=torch.int32, device="cuda"), residual)
    ref = _reference(z, mask, *params, residual=residual)
    torch.testing.assert_close(out[inside].float(), ref[inside], atol=ATOL, rtol=RTOL)
    assert torch.equal(out[~inside], z[~inside]), "rows outside n_valid must be left as they are"
    # flat rows [B, S, N, width]: every row computed, a 0 mask gives z with the residual and 0 without it; 5 rows
    # leave most warpgroups of the one tile without any
    for shape in ((2, 13, 51), (1, 1, 5)):
        z = torch.randn(*shape, width, device="cuda", generator=g).to(torch.bfloat16)
        mask = _random_mask(z.shape[:-1], width + 1) if with_mask else None
        out = _launch(width, hidden, z, mask, params, torch.full_like(z, float("nan")), None, residual)
        ref = _reference(z, mask, *params, residual=residual)
        torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
        if with_mask:
            masked = mask == 0
            assert torch.equal(out[masked], z[masked] if residual else torch.zeros_like(z[masked]))


@_requires_source
@pytest.mark.parametrize(("width", "hidden"), _RELU_SHAPES)
def test_every_relu_shape_with_a_large_mean(width, hidden):
    """|mean| >= 10 std at every width: the centred rows of the folded W1 keep z's mean out of GEMM 1."""
    skip_if_not_sm90()
    params = _shape_params(width, hidden, seed=3)
    g = torch.Generator(device="cuda").manual_seed(3)
    z = (-50.0 + 2.0 * torch.randn(3000, width, device="cuda", generator=g)).to(torch.bfloat16)
    out = _launch(width, hidden, z, None, params, torch.empty_like(z), None, True)
    ref = _reference(z, None, *params)
    torch.testing.assert_close(out.float(), ref, atol=_large_mean_atol(z), rtol=RTOL)


@_requires_source
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(("width", "hidden"), _RELU_SHAPES)
def test_every_relu_shape_persistent_image_matches_every_split_1_layout(width, hidden, bias):
    """The persistent image, its layout fixed while tracing, gives the bits of the other image's split-1 layouts at
    every tile height: padded pair maps in place and flat rows over several tiles per CTA, masks none, ones and with
    zeros, residual on and off."""
    skip_if_not_sm90()
    from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source

    kernel = _source.make_kernel(width, hidden, 1)
    tiles = [tile_wgs for tile_wgs, split in kernel.layouts if split == 1]
    params = _shape_params(width, hidden, seed=width + hidden + 5)
    if not bias:
        params = (*params[:3], None, params[4], None)
    g = torch.Generator(device="cuda").manual_seed(width + 5)
    lengths = torch.tensor([45, 70], dtype=torch.int32, device="cuda")
    pair = torch.randn(2, 70, 70, width, device="cuda", generator=g).to(torch.bfloat16)
    rows = 2 * torch.cuda.get_device_properties(0).multi_processor_count * kernel.tile_m + 37
    flat = torch.randn(rows, width, device="cuda", generator=g).to(torch.bfloat16)
    ones = tuple(torch.ones(shape, dtype=torch.bfloat16, device="cuda") for shape in (pair.shape[:-1], (rows,)))
    zeros = (_random_mask(pair.shape[:-1], width + 6), _random_mask((rows,), width + 7))
    for pair_mask, flat_mask in ((None, None), ones, zeros):
        for residual in (True, False):
            out = pair.clone()
            _launch(width, hidden, out, pair_mask, params, out, lengths, residual, persistent=True)
            flat_out = torch.full_like(flat, float("nan"))
            _launch(width, hidden, flat, flat_mask, params, flat_out, None, residual, persistent=True)
            for tile_wgs in tiles:
                got = pair.clone()
                _launch(width, hidden, got, pair_mask, params, got, lengths, residual, (tile_wgs, 1))
                assert torch.equal(got, out), f"pair maps at tile_wgs {tile_wgs}"
                got = torch.full_like(flat, float("nan"))
                _launch(width, hidden, flat, flat_mask, params, got, None, residual, (tile_wgs, 1))
                assert torch.equal(got, flat_out), f"flat rows at tile_wgs {tile_wgs}"


@_requires_source
@pytest.mark.parametrize(("width", "hidden", "num_wg"), [(64, 256, 2), (128, 512, 2)])
def test_chunk_ahead_schedule(width, hidden, num_wg):
    """GEMM 1 a chunk ahead (TILING's optional fourth field), over a resident and a streamed weight ring."""
    skip_if_not_sm90()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source
    from bionemo_ir._torch.utils.kernel import launch_compiled_kernel

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = _source.make_kernel(width, hidden, num_sms, overlap=True, num_wg=num_wg)
    exe = cute.compile(
        kernel,
        *_source.make_fake_args(width, hidden),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    params = _shape_params(width, hidden, seed=11)
    pack = w_norm.pack_weights(*params)
    g = torch.Generator(device="cuda").manual_seed(11)
    z = torch.randn(3, 37, 41, width, device="cuda", generator=g).to(torch.bfloat16)
    mask = _random_mask(z.shape[:-1], 12)
    for residual in (True, False):
        out = torch.full_like(z, float("nan"))
        launch_compiled_kernel(
            exe,
            z.view(-1),
            mask.data_ptr(),
            pack.image,
            pack.vec,
            out.view(-1),
            0,
            z.numel() // width,
            1,
            1,
            EPS,
            int(residual),
            1,
            num_wg,
        )
        ref = _reference(z, mask, *params, residual=residual)
        torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


# ---- the weight pack ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("width", "hidden"), [*_RELU_SHAPES, (384, 1536)])
def test_pack_folds_the_layernorm_and_holds_every_weight_once(width, hidden):
    """The image is W1 diag(gamma), rows centred, and W2, each chunk's rows in its own slots; the vector is
    ``b1 + W1 beta`` then ``b2``."""
    g = torch.Generator().manual_seed(0)

    def r(*shape, scale=1.0, offset=0.0):
        return (offset + scale * torch.randn(*shape, generator=g)).to(torch.bfloat16)

    gamma, beta = r(width, scale=0.1, offset=1.0), r(width, scale=0.1)
    w1, b1, w2, b2 = r(hidden, width), r(hidden), r(width, hidden), r(width)
    pack = w_norm.pack_weights(gamma, beta, w1, b1, w2, b2)
    assert pack.image.shape == (2 * width * hidden,) and pack.vec.shape == (hidden + width,)
    assert (pack.width, pack.hidden) == (width, hidden)
    expected_b1 = (b1.double() + w1.double() @ beta.double()).float()
    torch.testing.assert_close(pack.vec[:hidden], expected_b1, atol=0, rtol=0)
    assert torch.equal(pack.vec[hidden:], b2.float())
    folded = centre_rows_bf16(w1.double() * gamma.double())
    assert folded.float().sum(-1).abs().max() < 1e-2 * folded.float().abs().sum(-1).min()
    # per hidden chunk: the chunk's W1' rows, then its W2 columns
    chunk = w_norm.ops.IMAGE_LAYOUT["relu"][width][0]
    slots = pack.image.view(hidden // chunk, 2, -1)
    for j in range(hidden // chunk):
        rows = slice(chunk * j, chunk * (j + 1))
        assert torch.equal(slots[j, 0].float().sort().values, folded[rows].reshape(-1).float().sort().values)
        assert torch.equal(slots[j, 1].float().sort().values, w2[:, rows].reshape(-1).float().sort().values)
    # a missing bias is zero
    unbiased = w_norm.pack_weights(gamma, None, w1, None, w2, None)
    assert torch.equal(unbiased.vec, torch.zeros_like(unbiased.vec)) and torch.equal(unbiased.image, pack.image)


@_requires_source
def test_pack_layout_is_the_kernels():
    """pack_weights orders the image by the hidden chunk and W2 slots the kernel streams it with, and puts W1's
    columns in GEMM 1's fragment order wherever the kernel reads z from registers."""
    from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source

    layout, a_regs_width = w_norm.ops.IMAGE_LAYOUT, w_norm.ops.A_REGS_WIDTH
    tiling, kernel_a_regs_width = _source.kernel_tables()
    assert {act: set(widths) for act, widths in layout.items()} == {act: set(t) for act, t in tiling.items()}
    assert a_regs_width == kernel_a_regs_width
    for activation, widths in layout.items():
        for width, (chunk, w2_slots) in widths.items():
            for hidden in (2 * width, 4 * width):
                kernel = _source.make_kernel(width, hidden, 132, activation)
                assert (kernel.hc, kernel.p2) == (chunk, w2_slots), (activation, width, hidden)
                assert kernel.a_regs == (width <= a_regs_width.get(activation, 0)), (activation, width, hidden)


@pytest.mark.parametrize("width", [64, 128, 256])
def test_a_fragment_order_holds_each_threads_own_channels(width):
    """GEMM 1's K position k of fragment column ``2 q + e`` (+ 8) is a channel of quad lane q's 8-channel run."""
    order = w_norm.ops._a_fragment_order(width, torch.device("cpu"))
    assert sorted(order.tolist()) == list(range(width))
    k = torch.arange(width)
    lane = (k % 8) // 2  # the quad lane holding fragment column k % 16
    assert torch.equal((order % 32) // 8, lane)


def test_pack_rejects_other_shapes_and_operands():
    for width, hidden in ((192, 384), (128, 384), (512, 1024)):
        w1, w2 = torch.zeros(hidden, width, dtype=torch.bfloat16), torch.zeros(width, hidden, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="packs W1"):
            w_norm.pack_weights(torch.ones(width, dtype=torch.bfloat16), None, w1, None, w2, None)
    image, vec = torch.zeros(2 * C * H, dtype=torch.bfloat16), torch.zeros(H + C)
    w_norm.TransitionMlpWNormPack(image, vec, C, H)
    for bad in ((image.float(), vec), (image, vec.to(torch.bfloat16)), (image[:-8], vec), (image, vec[1:])):
        with pytest.raises(ValueError, match="pack_weights"):
            w_norm.TransitionMlpWNormPack(*bad, C, H)


# ---- the op, from source and from CUBINs ----------------------------------------------------------------------------

# Every (C, H, activation) the configs build on SM90, shipped or candidate.
_SHAPES = sorted(w_norm.shipped_shapes(90) | w_norm.candidate_shapes(90))
_SHAPE_IDS = [f"C{width}_H{hidden}_{activation}" for width, hidden, activation in _SHAPES]
_RESIDUAL = pytest.mark.parametrize("residual", [True, False], ids=["residual", "no_residual"])
_MASKED = pytest.mark.parametrize("masked", [True, False], ids=["mask", "no_mask"])


def _op_params(width: int, hidden: int, activation: str, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """gamma, beta, w1, b1, w2, b2 of one shape at the released weights' scale; SwiGLU's w1 holds the value rows, then
    the gate rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1_rows = (2 if activation == "silu_gate" else 1) * hidden

    def r(*shape, scale=1.0, offset=0.0):
        return (offset + scale * torch.randn(*shape, device="cuda", generator=g)).to(torch.bfloat16)

    return (
        r(width, scale=0.1, offset=1.0),
        r(width, scale=0.1),
        r(w1_rows, width, scale=width**-0.5),
        r(w1_rows, scale=0.1),
        r(width, hidden, scale=hidden**-0.5),
        r(width, scale=0.1),
    )


def _call(mode, monkeypatch, shape, params, z, mask, *, residual=True, n_valid=None, in_place=False):
    """The op through exactly one of the source and CUBIN paths: out of place into NaNs, or in place on a copy of
    ``z``."""
    if mode == "cubin":
        require_cubin_family("transition_mlp_w_norm")
    op = _op(*shape)
    pack = w_norm.pack_weights(*params, activation=shape[2])

    def call() -> torch.Tensor:
        out = z.clone() if in_place else torch.full_like(z, float("nan"))
        if in_place:
            return op(out, mask, pack, EPS, residual=residual, n_valid=n_valid, out=out)
        return op(z, mask, pack, EPS, residual=residual, n_valid=n_valid, out=out)

    return run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpWNormCuTe, w_norm_cutedsl, call)


def _shape_reference(shape, params, z, mask, *, residual):
    return w_norm.transition_mlp_w_norm_reference(z, mask, *params, EPS, activation=shape[2], residual=residual)


def _rows(shape: tuple[int, ...], width: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, width, device="cuda", generator=g).to(torch.bfloat16)


def _assert_masked_rows(out, z, mask, residual, rows=None):
    """A row whose mask is 0 comes out equal to z with the residual and exactly 0 without it."""
    masked = mask.reshape(z.shape[:-1]) == 0
    if rows is not None:
        masked &= rows
    expected = z[masked] if residual else torch.zeros_like(z[masked])
    assert torch.equal(out[masked], expected), "a row whose mask is 0 must come out equal to z, or 0 without z"


def _pair_problem(width, batch, tokens, lengths, seed):
    """Pair maps; a mask of 1, 0.5 and 0 over each item's valid rows and 0 elsewhere; and the valid rows themselves."""
    z = _rows((batch, tokens, tokens), width, seed)
    index = torch.arange(tokens, device="cuda")
    valid = torch.stack([(index[:, None] < n) & (index[None, :] < n) for n in lengths])
    mask = _random_mask(valid.shape, seed + 1) * valid
    return z, mask.unsqueeze(-1), valid


# Flat rows: one row, a partial last tile, an MSA's [B, S, N] rows and more tiles than SMs.
_FLAT_ROWS = [(1,), (1000,), (2, 7, 37), (2, 128, 136)]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
@_RESIDUAL
@_MASKED
@pytest.mark.parametrize("rows", _FLAT_ROWS, ids=["x".join(map(str, rows)) for rows in _FLAT_ROWS])
def test_op_flat_rows_match_fp32_reference(mode, shape, residual, masked, rows, monkeypatch):
    """Every row is computed; the mask, when there is one, scales the update."""
    skip_if_not_sm90()
    params = _op_params(*shape)
    z = _rows(rows, shape[0], 11)
    mask = _random_mask(rows, 12) if masked else None
    out = _call(mode, monkeypatch, shape, params, z, mask, residual=residual)
    expected = _shape_reference(shape, params, z, mask, residual=residual)
    torch.testing.assert_close(out.float(), expected, atol=ATOL, rtol=RTOL)
    if masked:
        _assert_masked_rows(out, z, mask, residual)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
@_RESIDUAL
@_MASKED
@pytest.mark.parametrize(
    ("batch", "tokens", "lengths"), [(1, 11, [7]), (3, 168, [168, 0, 50]), (2, 200, [9, 200]), (4, 72, [72, 1, 0, 40])]
)
def test_op_pair_maps_compute_their_valid_rows_in_place(
    mode, shape, residual, masked, batch, tokens, lengths, monkeypatch
):
    """With n_valid, the rows (b, i, j) with i, j < n_valid[b] are computed and the others keep z."""
    skip_if_not_sm90()
    params = _op_params(*shape, seed=1)
    z, mask, valid = _pair_problem(shape[0], batch, tokens, lengths, seed=13)
    mask = mask if masked else None
    n_valid = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    out = _call(mode, monkeypatch, shape, params, z, mask, residual=residual, n_valid=n_valid, in_place=True)
    expected = _shape_reference(shape, params, z, mask, residual=residual)
    torch.testing.assert_close(out[valid].float(), expected[valid], atol=ATOL, rtol=RTOL)
    assert torch.equal(out[~valid], z[~valid]), "rows outside n_valid must be left as they are"
    if masked:
        _assert_masked_rows(out, z, mask, residual, rows=valid)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
@pytest.mark.parametrize(("mean", "std"), [(20.0, 1.0), (-50.0, 2.0), (100.0, 1.0)])
def test_op_rows_with_a_large_mean(mode, shape, mean, std, monkeypatch):
    """|mean| >= 10 std: the centred rows of the folded W1 keep z's mean out of GEMM 1."""
    skip_if_not_sm90()
    params = _op_params(*shape, seed=3)
    g = torch.Generator(device="cuda").manual_seed(3)
    z = (mean + std * torch.randn(2, 40, 40, shape[0], device="cuda", generator=g)).to(torch.bfloat16)
    mask = torch.ones(2, 40, 40, device="cuda", dtype=torch.bfloat16)
    out = _call(mode, monkeypatch, shape, params, z, mask)
    expected = _shape_reference(shape, params, z, mask, residual=True)
    torch.testing.assert_close(out.float(), expected, atol=_large_mean_atol(z), rtol=RTOL)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(("batch", "tokens", "lengths"), [*_CASES, (4, 72, [72, 1, 0, 40])])
def test_op_matches_fp32_reference_and_passes_masked_rows_through(mode, batch, tokens, lengths, monkeypatch):
    """Pair maps as flat rows: every row is computed, and the mask keeps the padded ones equal to z."""
    skip_if_not_sm90()
    params = _params()
    z, mask = _problem(batch, tokens, lengths)
    out = _run_op(mode, monkeypatch, z, mask, params, out=torch.full_like(z, float("nan")))
    torch.testing.assert_close(out.float(), _reference(z, mask, *params), atol=ATOL, rtol=RTOL)
    masked = mask[..., 0] == 0
    assert torch.equal(out[masked], z[masked]), "a row whose mask is 0 must come out equal to z"


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
@_RESIDUAL
@_MASKED
@pytest.mark.parametrize("layout", ["flat", "pair"])
def test_cubin_matches_source_bit_for_bit(shape, residual, masked, layout, monkeypatch):
    """The packaged image is the source kernel: the same output, bit for bit, for every shipped shape, with and
    without the mask and the residual, over flat rows and over pair maps with their valid lengths."""
    skip_if_not_sm90()
    if set(_MODES) != {"source", "cubin"}:
        pytest.skip("needs BIOIR_TEST_CUTEDSL_MODES=source,cubin")
    params = _op_params(*shape, seed=4)
    if layout == "flat":
        z, n_valid = _rows((3, 96, 77), shape[0], 14), None
        mask = _random_mask((3, 96, 77), 15)
    else:
        lengths = [200, 131, 0]
        z, mask, _ = _pair_problem(shape[0], 3, 200, lengths, seed=16)
        n_valid = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    mask = mask if masked else None
    outputs = []
    for mode in ("source", "cubin"):
        with monkeypatch.context() as patch:
            outputs.append(
                _call(
                    mode, patch, shape, params, z, mask, residual=residual, n_valid=n_valid, in_place=layout == "pair"
                )
            )
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("lengths", [None, [56, 30]], ids=["flat", "pair"])
def test_op_replays_under_cuda_graph_capture(mode, lengths, monkeypatch):
    """A captured call reads the replay's z, mask and lengths."""
    skip_if_not_sm90()
    shape = _SHAPES[0]
    params = _op_params(*shape, seed=5)
    z, mask, _ = _pair_problem(shape[0], 2, 56, [56, 30], seed=5)
    n_valid = None if lengths is None else torch.tensor(lengths, dtype=torch.int32, device="cuda")
    expected = _call(mode, monkeypatch, shape, params, z, mask, n_valid=n_valid, in_place=True)
    op = _op(*shape)
    pack = w_norm.pack_weights(*params, activation=shape[2])
    static_z, static_mask = torch.zeros_like(z), torch.zeros_like(mask)
    static_lengths = None if n_valid is None else torch.zeros_like(n_valid)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(static_z, static_mask, pack, EPS, n_valid=static_lengths, out=static_z)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op(static_z, static_mask, pack, EPS, n_valid=static_lengths, out=static_z)
    torch.cuda.current_stream().wait_stream(stream)
    static_z.copy_(z)
    static_mask.copy_(mask)
    if static_lengths is not None:
        static_lengths.copy_(n_valid)
    graph.replay()
    assert torch.equal(static_z, expected)


# ---- launch layouts: smaller row tiles, and the hidden dimension split over a cluster of CTAs ------------------------


def _layout_calls(mode, monkeypatch, shape, params, z, mask, layouts, *, residual=True, n_valid=None, repeats=1):
    """Per layout ``(tile_wgs, split)``, the op's output ``repeats`` times through exactly one of the source and CUBIN
    paths: out of place into NaNs, or in place on a copy of ``z`` with ``n_valid``."""
    if mode == "cubin":
        require_cubin_family("transition_mlp_w_norm")
    op = _op(*shape)
    pack = w_norm.pack_weights(*params, activation=shape[2])
    outputs = {}

    def call() -> list[torch.Tensor]:
        results = []
        for layout in layouts:
            for _ in range(repeats):
                out = z.clone() if n_valid is not None else torch.full_like(z, float("nan"))
                source = out if n_valid is not None else z
                op(source, mask, pack, EPS, residual=residual, n_valid=n_valid, out=out, layout=layout)
                results.append(out)
        return results

    results = run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpWNormCuTe, w_norm_cutedsl, lambda: call())
    for index, layout in enumerate(layouts):
        outputs[layout] = results[index * repeats : (index + 1) * repeats]
    return outputs


def _full_layout(op) -> tuple[int, int]:
    return max(tile_wgs for tile_wgs, _ in op.layouts()), 1


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
@pytest.mark.parametrize("problem", ["flat", "pair", "full_pairs"])
def test_every_layout_matches_the_full_tile_or_repeats_its_split_sum(mode, shape, problem, monkeypatch):
    """Every layout the kernel takes: a split of 1, at any tile height, is the persistent full-tile launch bit for
    bit; a split sums its partials in a fixed order, so it repeats bit for bit, stays within bf16 rounding of the fp32
    reference, keeps masked rows exactly z (or 0) and leaves the rows outside n_valid alone. Full pair items round up
    to whole tiles each (70^2 rows are 38.3 tiles of 128), so the launch must hold a tile more than the rows alone."""
    skip_if_not_sm90()
    params = _op_params(*shape, seed=21)
    if problem == "flat":
        z, n_valid, valid, residual = _rows((1000,), shape[0], 22), None, None, True
        mask = _random_mask((1000,), 23)
    else:
        lengths = [70, 45] if problem == "pair" else [70, 70]
        z, mask, valid = _pair_problem(shape[0], 2, 70, lengths, seed=24)
        n_valid, residual = torch.tensor(lengths, dtype=torch.int32, device="cuda"), False
    layouts = _op(*shape).layouts()
    full = _full_layout(_op(*shape))
    assert full in layouts and any(split > 1 for _, split in layouts)
    outputs = _layout_calls(
        mode, monkeypatch, shape, params, z, mask, layouts, residual=residual, n_valid=n_valid, repeats=2
    )
    expected = _shape_reference(shape, params, z, mask, residual=residual)
    rows = torch.ones(z.shape[:-1], dtype=torch.bool, device="cuda") if valid is None else valid
    persistent = outputs[full][0]
    for layout, (out, again) in outputs.items():
        assert torch.equal(out, again), f"layout {layout} is not repeatable"
        if layout[1] == 1:
            assert torch.equal(out, persistent), f"split 1 at tile_wgs {layout[0]} differs from the full tile"
            continue
        torch.testing.assert_close(out[rows].float(), expected[rows], atol=ATOL, rtol=RTOL)
        _assert_masked_rows(out, z, mask, residual, rows=rows)
        if valid is not None:
            assert torch.equal(out[~valid], z[~valid]), f"layout {layout} touched rows outside n_valid"


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_cubin_matches_source_bit_for_bit_at_every_layout(shape, monkeypatch):
    """The packaged image splits and sums as the source kernel does: every layout, bit for bit."""
    skip_if_not_sm90()
    if set(_MODES) != {"source", "cubin"}:
        pytest.skip("needs BIOIR_TEST_CUTEDSL_MODES=source,cubin")
    params = _op_params(*shape, seed=25)
    z, mask = _rows((3, 7, 37), shape[0], 26), _random_mask((3, 7, 37), 27)
    layouts = _op(*shape).layouts()
    outputs = []
    for mode in ("source", "cubin"):
        with monkeypatch.context() as patch:
            outputs.append(_layout_calls(mode, patch, shape, params, z, mask, layouts))
    for layout in layouts:
        assert torch.equal(outputs[0][layout][0], outputs[1][layout][0]), layout


@pytest.mark.parametrize("mode", _MODES)
def test_op_takes_the_configured_layout_for_the_calls_waves(mode, monkeypatch):
    """The op launches the first configured layout whose waves (row tiles over the SM count) the call does not
    exceed, and the full tile past the last: a forced layout gives the same output bit for bit."""
    skip_if_not_sm90()
    shape = (C, H, "relu")
    params = _op_params(*shape, seed=28)
    z, mask = _rows((1000,), C, 29), _random_mask((1000,), 30)
    monkeypatch.setattr(TransitionMlpWNormCuTe, "_layout_cache", {})
    monkeypatch.setattr(w_norm._config, "layouts", lambda *args: ((1.0, 1, 2),))
    configured = _call(mode, monkeypatch, shape, params, z, mask)
    forced = _layout_calls(mode, monkeypatch, shape, params, z, mask, [(1, 2)])[(1, 2)][0]
    assert torch.equal(configured, forced)
    backend = _op(*shape).backend
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert backend.launch_layout(C, H, "relu", 128 * num_sms, 0) == (1, 2)  # one wave
    assert backend.launch_layout(C, H, "relu", 128 * num_sms + 1, 0) == (2, 1)
    with pytest.raises(ValueError, match="takes no layout tile_wgs=1, split=5"):
        _op(*shape)(z, mask, w_norm.pack_weights(*params), EPS, layout=(1, 5))


@pytest.mark.parametrize("mode", _MODES)
def test_op_runs_the_persistent_image_at_the_full_tile_and_the_other_at_every_other_layout(mode, monkeypatch):
    """A call whose layout is the full tile at split 1, chosen or forced, runs the shape's persistent image; any other
    layout runs the image that takes them all. Both give split 1 the same bits."""
    skip_if_not_sm90()
    shape = (C, H, "relu")
    params = _op_params(*shape, seed=40)
    launched = []
    launch, call = w_norm_cutedsl.launch_compiled_kernel, TransitionMlpWNormCubinExecutable.__call__
    monkeypatch.setattr(
        w_norm_cutedsl, "launch_compiled_kernel", lambda image, *args: launched.append(image) or launch(image, *args)
    )
    monkeypatch.setattr(
        TransitionMlpWNormCubinExecutable, "__call__", lambda image, *args: launched.append(image) or call(image, *args)
    )
    monkeypatch.setattr(TransitionMlpWNormCuTe, "_layout_cache", {})
    backend = TransitionMlpWNormCuTe(90)
    small = _rows((1000,), C, 41)  # 8 tiles: a split layout
    large = _rows((40000,), C, 42)  # past every configured layout: the full tile at split 1
    for z, layouts, kinds in (
        (small, [(2, 1), (1, 1)], [True, False]),
        (large, [None], [True]),
        (small, [None], [False]),
    ):
        launched.clear()
        outputs = _layout_calls(mode, monkeypatch, shape, params, z, None, layouts)
        cache = TransitionMlpWNormCuTe._compiled_cache  # this mode's, as the calls left it
        images = {
            id(cache[key]): persistent
            for persistent in (False, True)
            if (key := backend._key(C, H, "relu", persistent)) in cache
        }
        assert [images[id(image)] for image in launched] == kinds * 2  # the calls run twice
        if len(layouts) == 2:
            assert torch.equal(outputs[(2, 1)][0], outputs[(1, 1)][0])


def test_config_layouts_are_the_kernels():
    """Every layout a config entry lists is one its kernel takes."""
    skip_if_not_sm90()
    for width, hidden, activation in _SHAPES:
        allowed = set(_op(width, hidden, activation).layouts())
        for _, tile_wgs, split in w_norm._config.layouts(90, width, hidden, activation):
            assert (tile_wgs, split) in allowed, (width, hidden, activation, tile_wgs, split)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("layout", [(1, 4), (2, 2), (2, 1)], ids=["w1s4", "w2s2", "w2s1"])
def test_split_layouts_replay_under_cuda_graph_capture(mode, layout, monkeypatch):
    """A captured split launch keeps no state between replays: each replay of new inputs matches the eager call."""
    skip_if_not_sm90()
    shape = (C, H, "relu")
    params = _op_params(*shape, seed=31)
    problems = [_pair_problem(C, 2, 40, [40, 23], seed=seed) for seed in (32, 33)]
    n_valid = torch.tensor([40, 23], dtype=torch.int32, device="cuda")
    expected = [
        _layout_calls(mode, monkeypatch, shape, params, z, mask, [layout], n_valid=n_valid)[layout][0]
        for z, mask, _ in problems
    ]
    op = _op(*shape)
    pack = w_norm.pack_weights(*params)
    static_z, static_mask = torch.zeros_like(problems[0][0]), torch.zeros_like(problems[0][1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(static_z, static_mask, pack, EPS, n_valid=n_valid, out=static_z, layout=layout)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op(static_z, static_mask, pack, EPS, n_valid=n_valid, out=static_z, layout=layout)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        for (z, mask, _), want in zip(problems, expected, strict=True):
            static_z.copy_(z)
            static_mask.copy_(mask)
            graph.replay()
            assert torch.equal(static_z, want)


@pytest.mark.parametrize("mode", _MODES)
def test_split_layouts_run_on_two_streams_at_once(mode, monkeypatch):
    """Two streams launch split layouts at the same time, each on its own inputs: they share no workspace or counter,
    so each stream's output is its own call's."""
    skip_if_not_sm90()
    shape = (C, H, "relu")
    params = _op_params(*shape, seed=34)
    problems = [(_rows((1500,), C, seed), _random_mask((1500,), seed + 1)) for seed in (35, 37)]
    layouts = [(1, 4), (1, 3)]
    expected = [
        _layout_calls(mode, monkeypatch, shape, params, z, mask, [layout])[layout][0]
        for (z, mask), layout in zip(problems, layouts, strict=True)
    ]

    def run_both() -> list[torch.Tensor]:
        op = _op(*shape)
        pack = w_norm.pack_weights(*params)
        outs = [torch.full_like(z, float("nan")) for z, _ in problems]
        streams = [torch.cuda.Stream() for _ in problems]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for _ in range(20):  # back to back on both streams, so the launches overlap
            for stream, (z, mask), out, layout in zip(streams, problems, outs, layouts, strict=True):
                with torch.cuda.stream(stream):
                    op(z, mask, pack, EPS, out=out, layout=layout)
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        return outs

    outs = run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpWNormCuTe, w_norm_cutedsl, run_both)
    for out, want in zip(outs, expected, strict=True):
        assert torch.equal(out, want)


def test_op_accepts_contiguous_bf16_rows_and_pair_maps_with_their_lengths():
    skip_if_not_sm90()
    op = _op()

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(*shape, device="cuda", dtype=dtype)

    z = zeros(2, 9, 9, C)
    mask = torch.ones(2, 9, 9, 1, device="cuda", dtype=torch.bfloat16)
    n_valid = torch.tensor([9, 4], dtype=torch.int32, device="cuda")
    # flat rows of any leading shape, with or without a mask
    assert op.accepts(z, mask) and op.accepts(z, mask[..., 0]) and op.accepts(z, None)
    assert op.accepts(zeros(100, C), None) and op.accepts(zeros(2, 5, 9, C), mask[:, :5, :, 0].contiguous())
    assert op.accepts(zeros(65, 1, 1, C), None)
    # pair maps with their lengths
    assert op.accepts(z, mask, n_valid=n_valid) and op.accepts(z, None, n_valid=n_valid)
    assert not op.accepts(zeros(2, 5, 9, C), None, n_valid=n_valid)  # not square
    assert not op.accepts(zeros(2, 9, C), None, n_valid=n_valid)  # not a pair map
    assert not op.accepts(z, mask, n_valid=n_valid.long())
    assert not op.accepts(z, mask, n_valid=n_valid[:1])
    assert not op.accepts(z, mask, n_valid=n_valid.cpu())
    assert not op.accepts(zeros(65, 1, 1, C), None, n_valid=torch.ones(65, dtype=torch.int32, device="cuda"))
    # z and mask
    assert not op.accepts(z.float(), mask)
    assert not op.accepts(z.transpose(1, 2), mask)
    assert not op.accepts(z[:, :5], mask[:, :5])  # a chunk of rows that is not contiguous
    assert not op.accepts(zeros(2, 9, 9, C // 2), mask)
    assert not op.accepts(zeros(0, C), None)
    assert not op.accepts(zeros(C), None)
    assert op.accepts(zeros(10 * C + 8)[8:].view(10, C), None)  # 16 bytes past 16-byte alignment is fine
    assert not op.accepts(zeros(10 * C + 1)[1:].view(10, C), None)  # 2 bytes past it is not
    assert not op.accepts(z, mask.float())
    assert not op.accepts(z, mask.cpu())
    assert not op.accepts(z, mask[:1])
    assert not op.accepts(z, torch.ones(2, 9, 9, 2, device="cuda", dtype=torch.bfloat16)[..., :1])
    # z on a device of another SM than the op's backend
    sm80_op = w_norm.TransitionMlpWNormOp(TransitionMlpWNormCuTe(80), op.width, op.hidden, op.activation)
    assert not sm80_op.accepts(z, mask)


def test_op_takes_out_in_place_or_apart_from_every_operand():
    skip_if_not_sm90()
    op = _op()
    z = torch.zeros(2, 9, 9, C, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 9, 9, device="cuda", dtype=torch.bfloat16)
    pack = w_norm.pack_weights(*_params())
    buffer = torch.zeros(2 * z.numel(), device="cuda", dtype=torch.bfloat16)
    shifted = buffer[C * 8 : C * 8 + z.numel()].view(z.shape)
    with pytest.raises(ValueError, match="must be z itself or not overlap it"):
        op(buffer[: z.numel()].view(z.shape), mask, pack, EPS, out=shifted)
    with pytest.raises(ValueError, match="must not overlap the mask"):
        op(z, buffer[: mask.numel()].view(mask.shape), pack, EPS, out=buffer[: z.numel()].view(z.shape))
    with pytest.raises(ValueError, match="contiguous, 16-byte aligned bf16"):
        op(z, mask, pack, EPS, out=torch.empty_like(z).transpose(1, 2))
    with pytest.raises(ValueError, match="needs contiguous bf16"):
        op(z.float(), mask, pack, EPS)
    gated = w_norm.TransitionMlpWNormPack(
        torch.zeros(3 * C * H, device="cuda", dtype=torch.bfloat16),
        torch.zeros(2 * H + C, device="cuda"),
        C,
        H,
        "silu_gate",
    )
    with pytest.raises(ValueError, match="activation='relu' pack"):
        op(z, mask, gated, EPS)
    assert op(z, mask, pack, EPS, out=z) is z


def test_op_ships_where_a_config_entry_declares_it():
    """A shipped entry serves every caller; a candidate entry only a caller that asks for candidates."""
    skip_if_not_sm90()
    assert (C, H, "relu") in w_norm.shipped_shapes(90)
    assert w_norm.get_transition_mlp_w_norm_op(torch.float32, C, H) is None
    assert w_norm.get_transition_mlp_w_norm_op(torch.bfloat16, 384, 1536, candidates=True) is None
    assert w_norm.get_transition_mlp_w_norm_op(torch.bfloat16, C, H, "gelu", candidates=True) is None
    for width, hidden, activation in _SHAPES:
        shipped = (width, hidden, activation) in w_norm.shipped_shapes(90)
        op = w_norm.get_transition_mlp_w_norm_op(torch.bfloat16, width, hidden, activation)
        assert (op is not None) == shipped, (width, hidden, activation)
        op = w_norm.get_transition_mlp_w_norm_op(torch.bfloat16, width, hidden, activation, candidates=True)
        assert op is not None and (op.width, op.hidden, op.activation) == (width, hidden, activation)


def test_launcher_checks_spans_alignment_devices_and_aliasing():
    """The C++ launcher refuses what the op would: an extent, alignment or device that does not match, an output that
    partly overlaps x or overlaps any operand it reads, and a pair map whose shape or lengths disagree with rows."""
    skip_if_not_sm90()
    if "cubin" not in _MODES:
        pytest.skip("needs the CUBIN library: BIOIR_TEST_CUTEDSL_MODES=cubin")
    require_cubin_family("transition_mlp_w_norm")
    from bionemo_ir._torch.utils.kernel import current_stream_handle, tensor_flat, tensor_s1_d0

    library = library_runtime._load_kernel_library()
    launcher = library.transition_mlp_w_norm
    config = launcher.make_kernel_config(90, launcher.DType.BFLOAT16, C, H, launcher.Activation.RELU)
    pack = w_norm.pack_weights(*_params(17))
    batch, tokens = 2, 9
    rows = batch * tokens * tokens
    buffer = torch.zeros(3 * rows * C, device="cuda", dtype=torch.bfloat16)
    z, out = buffer[: rows * C], buffer[rows * C : 2 * rows * C]
    mask = torch.ones(rows, device="cuda", dtype=torch.bfloat16)
    n_valid = torch.full((batch,), tokens, dtype=torch.int32, device="cuda")

    def params(**override):
        values = {"x": z, "mask": mask, "image": pack.image, "vec": pack.vec, "output": out, "n_valid": None}
        values.update({key: value for key, value in override.items() if key in values})
        launch = launcher.LaunchParams()
        for name in ("x", "mask", "image", "output"):
            if values[name] is not None:
                setattr(launch, name, tensor_flat(library, values[name]))
        launch.vec = tensor_s1_d0(library, values["vec"])
        if values["n_valid"] is not None:
            launch.n_valid = tensor_s1_d0(library, values["n_valid"])
        launch.rows = override.get("rows", rows)
        launch.n = override.get("n", tokens)
        launch.n_batch = override.get("n_batch", batch)
        launch.eps = EPS
        launch.residual = True
        launch.split = override.get("split", 1)
        launch.tile_wgs = override.get("tile_wgs", 0)
        launch.stream = current_stream_handle(z)
        return launch

    launcher.launch(config, params())  # flat rows, out of place
    launcher.launch(config, params(output=z))  # in place
    launcher.launch(config, params(n_valid=n_valid, mask=None))  # pair maps, no mask
    launcher.launch(config, params(split=4, tile_wgs=1))  # a cluster of 4 CTAs per 64-row tile
    launcher.launch(config, params(split=2, tile_wgs=0))  # every warpgroup
    persistent = launcher.make_kernel_config(90, launcher.DType.BFLOAT16, C, H, launcher.Activation.RELU, True)
    assert persistent.spec.is_persistent and not config.spec.is_persistent
    launcher.launch(persistent, params())  # the full tile at split 1: the persistent image's only layout
    launcher.launch(persistent, params(tile_wgs=2))
    torch.cuda.synchronize()
    assert launcher.supports_layout(persistent.spec, 1, 0) and not launcher.supports_layout(persistent.spec, 1, 1)
    for split, tile_wgs in ((2, 0), (1, 1)):
        with pytest.raises(ValueError, match="takes no layout"):
            launcher.launch(persistent, params(split=split, tile_wgs=tile_wgs))
    assert launcher.supports_layout(config.spec, 6, 2) and not launcher.supports_layout(config.spec, 5, 1)
    for split, tile_wgs in ((5, 1), (0, 1), (9, 2), (1, 4), (2, 3)):
        with pytest.raises(ValueError, match="takes no layout"):
            launcher.launch(config, params(split=split, tile_wgs=tile_wgs))
    for override, match in (
        ({"rows": 0}, "between 1 and"),
        ({"rows": 2**31 - 256}, "between 1 and"),
        ({"x": z[:-C]}, "x must hold"),
        ({"mask": mask[:-1]}, "mask must hold"),
        ({"image": pack.image[:-8]}, "image must hold"),
        ({"vec": pack.vec[:-1]}, "vec must hold"),
        ({"x": buffer[1 : rows * C + 1]}, "alignment"),
        ({"mask": mask.cpu()}, "no CUDA device"),
        ({"output": buffer[rows * C // 2 : rows * C // 2 + rows * C]}, "x itself or not overlap"),
        ({"mask": out[:rows]}, "must not overlap mask"),
        ({"n_valid": n_valid, "rows": rows - 1}, "rows == n_batch \\* n \\* n"),
        ({"n_valid": n_valid, "n_batch": 65, "rows": 65 * tokens * tokens}, "between 1 and 64 pair maps"),
        ({"n_valid": n_valid[:1]}, "one length per pair map"),
        ({"n_valid": out[: 2 * batch].view(torch.int32)}, "must not overlap n_valid"),
    ):
        with pytest.raises(ValueError, match=match):
            launcher.launch(config, params(**override))
    with pytest.raises(ValueError, match="No embedded fused transition CUBIN"):
        launcher.make_kernel_config(90, launcher.DType.BFLOAT16, 384, 1536, launcher.Activation.RELU)


class _LauncherBeforeTheActivationAxis:
    """The first launch ABI's launcher: no activation argument, and a kernel that needs the mask and the residual."""

    class DType:
        BFLOAT16 = 0

    @staticmethod
    def make_kernel_config(target_sm, dtype, width, hidden):
        return object()


class _LauncherBeforeTheLaunchLayout(_LauncherBeforeTheActivationAxis):
    """The second launch ABI's launcher: the activation axis, and a kernel without the launch layout arguments."""

    class Activation:
        RELU = 0
        SILU_GATE = 1

    @staticmethod
    def make_kernel_config(target_sm, dtype, width, hidden, activation):
        return object()


@pytest.mark.parametrize("launcher", [_LauncherBeforeTheActivationAxis, _LauncherBeforeTheLaunchLayout])
def test_op_does_not_ship_from_a_library_before_its_launch_abi(launcher, monkeypatch):
    """A kernel library built before the activation axis cannot run a mask-free or residual-free call, and one built
    before the launch layouts cannot take a split or a smaller tile, so nothing ships from either."""
    with pytest.raises(CuTeDSLKernelVariantUnavailable, match="predates launch ABI"):
        TransitionMlpWNormCubinExecutable(None, launcher, 90, C, H, "relu")
    monkeypatch.setattr(library_runtime, "_kernel_library", SimpleNamespace(transition_mlp_w_norm=launcher))
    monkeypatch.setattr(TransitionMlpWNormCuTe, "_compiled_cache", {})
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    assert not TransitionMlpWNormCuTe(90).ships(C, H, "relu")


def test_forced_cubin_does_not_ship_from_a_cached_source_build(monkeypatch):
    """Under CUTEDSL_FORCE_CUBIN a source executable cached earlier in the process does not stand in for the CUBIN:
    the library must hold one."""
    backend = TransitionMlpWNormCuTe(90)
    key = backend._key(C, H, "relu")
    monkeypatch.setattr(library_runtime, "_kernel_library", SimpleNamespace())
    monkeypatch.setattr(TransitionMlpWNormCuTe, "_compiled_cache", {key: object()})
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    assert not backend.ships(C, H, "relu")
    assert key not in TransitionMlpWNormCuTe._compiled_cache


@pytest.mark.parametrize(
    ("z_device", "current", "expected"),
    [(1, 0, ["enter cuda:1", "compile", "launch", "exit"]), (0, 0, ["compile", "launch"])],
)
def test_run_makes_the_operands_device_current_before_compiling(monkeypatch, z_device, current, expected):
    """A call on another GPU than the current one compiles, loads and launches with z's device current; a call on
    the current GPU switches nothing."""
    events = []

    @contextlib.contextmanager
    def device(target):
        events.append(f"enter {target}")
        yield
        events.append("exit")

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(w_norm_cutedsl, "launch_compiled_kernel", lambda *args: events.append("launch"))
    backend = TransitionMlpWNormCuTe(90)
    monkeypatch.setattr(backend, "_layout", lambda *args: (2, 1, True))
    monkeypatch.setattr(backend, "_executable", lambda *args: events.append("compile") or "source executable")
    z = SimpleNamespace(get_device=lambda: z_device, device=torch.device("cuda", z_device))
    backend.run(z, None, None, None, None, 1, 0, 0, EPS, True, None, C, H, "relu")
    assert events == expected
