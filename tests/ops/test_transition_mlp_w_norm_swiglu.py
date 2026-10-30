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
"""SwiGLU (``silu(gate) * value``) in the fused LayerNorm transition kernel: every SwiGLU shape on padded pair maps
and on flat rows, with and without the mask and the residual, against an fp32 reference; and the pack that folds the
LayerNorm into both halves of W1."""

from __future__ import annotations

import pytest
import torch

from bionemo_ir._torch.custom_ops import transition_mlp_w_norm as w_norm
from bionemo_ir._torch.custom_ops.trimul_kf_k3.ops import centre_rows_bf16
from tests._torch import skip_if_not_sm90, source_module_available

_SOURCE = "bionemo_ir._torch.custom_ops.transition_mlp_w_norm._source"
_requires_source = pytest.mark.skipif(not source_module_available(_SOURCE), reason="launches the kernel from source")
EPS = 1e-5
ATOL, RTOL = 5e-2, 2e-2  # tests/ops/test_transition_mlp.py's tolerance
# (C, H) of every SwiGLU transition in scope; each runs on pair maps and on flat rows
SHAPES = [(128, 512), (256, 1024), (64, 256), (64, 128), (128, 256), (256, 512)]
_EXE = {}


def _compiled(width, hidden, persistent=False):
    """The SwiGLU kernel compiled from source through the kernel disk cache, which can be seeded outside a GPU lock:
    the image of every layout, or the persistent image (``persistent``)."""
    if (width, hidden, persistent) not in _EXE:
        from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source
        from bionemo_ir.dsl_kernels.cute_cache import CuteKernelCache

        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        cache = CuteKernelCache()
        key = ("test_transition_mlp_w_norm_swiglu_v3", width, hidden, persistent, num_sms)
        executable = cache.load_from_cache(key)
        if executable is None:
            executable = _source.compile_transition_mlp_w_norm_source(
                cache.compile, width, hidden, num_sms, "silu_gate", persistent=persistent
            )
            cache.save_to_cache(key, executable)
        _EXE[(width, hidden, persistent)] = executable
    return _EXE[(width, hidden, persistent)]


def _params(width, hidden, seed, bias=True, device="cuda"):
    """gamma, beta, w1 ``[2 H, C]`` (value rows, then gate rows), b1, w2, b2 at the released weights' scale."""
    g = torch.Generator(device=device).manual_seed(seed)

    def r(*shape, scale=1.0, offset=0.0):
        return (offset + scale * torch.randn(*shape, device=device, generator=g)).to(torch.bfloat16)

    gamma, beta = r(width, scale=0.1, offset=1.0), r(width, scale=0.1)
    w1, b1 = r(2 * hidden, width, scale=width**-0.5), r(2 * hidden, scale=0.1)
    w2, b2 = r(width, hidden, scale=hidden**-0.5), r(width, scale=0.1)
    return (gamma, beta, w1, b1 if bias else None, w2, b2 if bias else None)


def _reference(z, mask, params, residual):
    """fp32 ``z + mask * u`` with the residual, ``mask * u`` without it; ``mask`` ``None`` keeps every row."""
    ones = torch.ones(z.shape[:-1], dtype=torch.float32, device=z.device)
    out = w_norm.transition_mlp_w_norm_reference(
        z, ones if mask is None else mask, *params, EPS, activation="silu_gate"
    )
    return out if residual else out - z.float()


def _launch(z, mask, pack, out, n_valid, residual, layout=None, persistent=False):
    """Flat rows when ``n_valid`` is None, else pair maps ``[B, N, N, C]`` over their valid rows. ``layout``:
    ``(tile_wgs, split)``, by default the persistent launch; ``persistent``: the persistent image (it takes the full
    tile at split 1 only)."""
    from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source
    from bionemo_ir._torch.utils.kernel import launch_compiled_kernel

    width = z.shape[-1]
    batch, tokens = (1, 1) if n_valid is None else (z.shape[0], z.shape[1])
    tile_wgs, split = layout or (_source.launch_geometry(width, pack.hidden, "silu_gate")["tile_m"] // 64, 1)
    launch_compiled_kernel(
        _compiled(width, pack.hidden, persistent),
        *(z.view(-1), 0 if mask is None else mask.data_ptr(), pack.image, pack.vec, out.view(-1)),
        *(0 if n_valid is None else n_valid.data_ptr(), z.numel() // width, tokens, batch, EPS, int(residual)),
        *(split, tile_wgs),
    )
    return out


def _random_mask(shape, seed):
    """Mostly 1, with 0.5 and 0 rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    picks = torch.rand(shape, device="cuda", generator=g)
    return torch.where(picks < 0.1, 0.5, torch.where(picks < 0.2, 0.0, 1.0)).to(torch.bfloat16)


@_requires_source
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize(("width", "hidden"), SHAPES)
def test_every_shape_on_pair_maps_and_flat_rows(width, hidden, with_mask, residual):
    """Padded pair maps in place (rows outside n_valid untouched) and flat MSA-like rows, against fp32: a row whose
    mask is 0 comes out exactly z with the residual and exactly 0 without it."""
    skip_if_not_sm90()
    params = _params(width, hidden, seed=width + hidden, bias=with_mask)
    pack = w_norm.pack_weights(*params, activation="silu_gate")
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
    _launch(out, mask, pack, out, torch.tensor(lengths, dtype=torch.int32, device="cuda"), residual)
    ref = _reference(z, mask, params, residual)
    torch.testing.assert_close(out[inside].float(), ref[inside], atol=ATOL, rtol=RTOL)
    assert torch.equal(out[~inside], z[~inside]), "rows outside n_valid must be left as they are"
    if with_mask:
        dropped = inside & (mask == 0)
        assert torch.equal(out[dropped], z[dropped] if residual else torch.zeros_like(z[dropped]))
    # flat rows [B, S, N, C]: every row computed
    z = torch.randn(2, 13, 51, width, device="cuda", generator=g).to(torch.bfloat16)
    mask = _random_mask(z.shape[:-1], width + 1) if with_mask else None
    out = _launch(z, mask, pack, torch.full_like(z, float("nan")), None, residual)
    torch.testing.assert_close(out.float(), _reference(z, mask, params, residual), atol=ATOL, rtol=RTOL)
    if with_mask:
        dropped = mask == 0
        assert torch.equal(out[dropped], z[dropped] if residual else torch.zeros_like(z[dropped]))


@_requires_source
@pytest.mark.parametrize(("width", "hidden"), SHAPES)
def test_every_shape_with_a_large_mean(width, hidden):
    """|mean| >= 10 std: the centred bf16 rows of both W1 halves keep z's mean out of GEMM 1."""
    skip_if_not_sm90()
    params = _params(width, hidden, seed=3)
    pack = w_norm.pack_weights(*params, activation="silu_gate")
    g = torch.Generator(device="cuda").manual_seed(3)
    z = (-50.0 + 2.0 * torch.randn(3000, width, device="cuda", generator=g)).to(torch.bfloat16)
    out = _launch(z, None, pack, torch.empty_like(z), None, True)
    atol = float(ATOL + 0.5 * 2.0 ** (torch.floor(torch.log2(z.float().abs().max())) - 7))
    torch.testing.assert_close(out.float(), _reference(z, None, params, True), atol=atol, rtol=RTOL)


@_requires_source
def test_is_deterministic_and_flat_rows_match_the_pair_layout():
    skip_if_not_sm90()
    width, hidden = SHAPES[0]
    pack = w_norm.pack_weights(*_params(width, hidden, seed=5), activation="silu_gate")
    g = torch.Generator(device="cuda").manual_seed(5)
    z = torch.randn(2, 72, 72, width, device="cuda", generator=g).to(torch.bfloat16)
    mask = _random_mask(z.shape[:-1], 5)
    full = torch.full((2,), 72, dtype=torch.int32, device="cuda")
    first = _launch(z, mask, pack, torch.empty_like(z), full, True)
    second = _launch(z, mask, pack, torch.empty_like(z), full, True)
    flat = _launch(z, mask, pack, torch.empty_like(z), None, True)
    assert torch.equal(first, second) and torch.equal(first, flat)


@_requires_source
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(("width", "hidden"), SHAPES)
def test_every_shape_persistent_image_matches_every_split_1_layout(width, hidden, bias):
    """The persistent image, its layout fixed while tracing, gives the bits of the other image's split-1 layouts at
    every tile height: padded pair maps in place and flat rows over several tiles per CTA, masks none, ones and with
    zeros, residual on and off."""
    skip_if_not_sm90()
    from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import _source

    kernel = _source.make_kernel(width, hidden, 1, "silu_gate")
    tiles = [tile_wgs for tile_wgs, split in kernel.layouts if split == 1]
    params = _params(width, hidden, seed=width + hidden + 5, bias=bias)
    pack = w_norm.pack_weights(*params, activation="silu_gate")
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
            _launch(out, pair_mask, pack, out, lengths, residual, persistent=True)
            flat_out = _launch(
                flat, flat_mask, pack, torch.full_like(flat, float("nan")), None, residual, persistent=True
            )
            for tile_wgs in tiles:
                got = pair.clone()
                _launch(got, pair_mask, pack, got, lengths, residual, (tile_wgs, 1))
                assert torch.equal(got, out), f"pair maps at tile_wgs {tile_wgs}"
                got = _launch(flat, flat_mask, pack, torch.full_like(flat, float("nan")), None, residual, (tile_wgs, 1))
                assert torch.equal(got, flat_out), f"flat rows at tile_wgs {tile_wgs}"


# ---- the pack -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("width", "hidden"), SHAPES)
def test_pack_folds_both_halves_and_holds_every_weight_once(width, hidden):
    """Per hidden chunk: the W1' slots hold the chunk's value rows and its gate rows halved, split along C; the W2
    slots its hidden columns. The vector is b1 + W1 beta with the gate half halved, then b2."""
    gamma, beta, w1, b1, w2, b2 = _params(width, hidden, seed=13, device="cpu")
    pack = w_norm.pack_weights(gamma, beta, w1, b1, w2, b2, activation="silu_gate")
    assert pack.image.shape == (3 * width * hidden,) and pack.vec.shape == (2 * hidden + width,)
    assert (pack.width, pack.hidden, pack.activation) == (width, hidden, "silu_gate")
    folded_bias = (b1.double() + w1.double() @ beta.double()).float()
    torch.testing.assert_close(pack.vec[:hidden], folded_bias[:hidden], atol=0, rtol=0)
    torch.testing.assert_close(pack.vec[hidden : 2 * hidden], 0.5 * folded_bias[hidden:], atol=0, rtol=0)
    assert torch.equal(pack.vec[2 * hidden :], b2.float())
    folded = centre_rows_bf16(w1.double() * gamma.double())
    value, gate = folded[:hidden], folded[hidden:] * 0.5
    chunk, w2_slots = w_norm.ops.IMAGE_LAYOUT["silu_gate"][width]
    w1_slots = 2 * w2_slots
    slots = pack.image.view(hidden // chunk, w1_slots + w2_slots, -1)

    def same(slot, block):
        return torch.equal(slot.float().sort().values, block.reshape(-1).float().sort().values)

    for j in range(hidden // chunk):
        units = slice(chunk * j, chunk * (j + 1))
        rows = torch.cat([value[units], gate[units]])
        k1, k2 = width // w1_slots, chunk // w2_slots
        for p in range(w1_slots):
            assert same(slots[j, p], rows[:, k1 * p : k1 * (p + 1)]), (j, p)
        for p in range(w2_slots):
            assert same(slots[j, w1_slots + p], w2[:, units][:, k2 * p : k2 * (p + 1)]), (j, p)


def test_pack_rejects_other_shapes_and_activations():
    width, hidden = SHAPES[0]
    gamma, beta, w1, b1, w2, b2 = _params(width, hidden, seed=14, device="cpu")
    relu_sized = w_norm.pack_weights(gamma, beta, w1[:hidden], b1[:hidden], w2, b2)
    with pytest.raises(ValueError, match="pack_weights"):
        w_norm.TransitionMlpWNormPack(relu_sized.image, relu_sized.vec, width, hidden, "silu_gate")
    with pytest.raises(ValueError, match="packs W1"):
        w_norm.pack_weights(gamma, beta, w1[:hidden], b1[:hidden], w2, b2, activation="silu_gate")
    with pytest.raises(ValueError, match="packs W1"):
        w_norm.pack_weights(gamma, beta, w1, b1, w2, b2, activation="gelu")
    big = _params(384, 768, seed=15, device="cpu")
    with pytest.raises(ValueError, match="packs W1"):
        w_norm.pack_weights(*big, activation="silu_gate")
