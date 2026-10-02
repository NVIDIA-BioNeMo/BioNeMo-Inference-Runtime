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
"""Token-count padding for model trunks and confidence pairformers.

Model configs name the tensors each region pads with ``TrunkPadSpec`` and
``FeatureDictPadSpec`` (``bionemo_ir.configs``); this module applies them.

``TriangleMultiplicationNode``'s fused residual epilogue
(``can_fuse_residual``, ``triangle_nodes.py``) requires both pair token dims
to be a multiple of ``TOKEN_ALIGN``. Real sequence lengths essentially never
are, so every trimul layer in every block of a trunk's layer stack silently
falls back to an unfused, much slower residual add. Padding once per trunk
forward call, before its recycling/layer loop, instead of leaving each layer
to discover the misalignment on its own, covers every sub-module the trunk
runs (MSA module, template embedder, pairformer stack, ...) in one pass.

Padding is always appended at the end (never inserted), which preserves the
left-aligned ``1...1 0...0`` mask convention CuTeDSL's prefix-length masking
requires. Always zeros, never arbitrary values: downstream epilogues
multiply by the mask and ``inf * 0 == nan``, so an uninitialised gap would
propagate even through masked paths.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bionemo_ir.configs import FeatureDictPadSpec, TrunkPadSpec

# Matches triangle_nodes._GEMM_TOKEN_ALIGN.
TOKEN_ALIGN = 8


def round_up_tokens(n: int, align: int = TOKEN_ALIGN) -> int:
    return -(-n // align) * align


def pad_single_token_dim(t: torch.Tensor, pad: int, *, has_channel: bool) -> torch.Tensor:
    """Zero-pad one trailing token dim by ``pad`` on the right.

    ``has_channel=True`` for ``[..., N, C]`` tensors (pad the N dim, one
    before the end); ``False`` for ``[..., N]`` tensors (pad the last dim).
    """
    f_pad = [0, 0, 0, pad] if has_channel else [0, pad]
    return F.pad(t, f_pad)


def pad_pair_token_dims(t: torch.Tensor, pad: int, *, has_channel: bool) -> torch.Tensor:
    """Zero-pad two trailing token dims (``[..., N, N, ...]``) by ``pad`` each.

    ``has_channel=True`` for ``[..., N, N, C]`` tensors (pad the two N dims,
    leaving a trailing channel dim alone); ``False`` for ``[..., N, N]``
    tensors (pad the two trailing dims directly).
    """
    f_pad = [0, 0, 0, pad, 0, pad] if has_channel else [0, pad, 0, pad]
    return F.pad(t, f_pad)


def pad_feature_dict(d: dict, pad: int, spec: FeatureDictPadSpec) -> dict:
    out = dict(d)
    for key in spec.single_last:
        if out.get(key) is not None:
            out[key] = pad_single_token_dim(out[key], pad, has_channel=False)
    for key in spec.single_channel:
        if out.get(key) is not None:
            out[key] = pad_single_token_dim(out[key], pad, has_channel=True)
    for key in spec.pair_last:
        if out.get(key) is not None:
            out[key] = pad_pair_token_dims(out[key], pad, has_channel=False)
    for key in spec.pair_channel:
        if out.get(key) is not None:
            out[key] = pad_pair_token_dims(out[key], pad, has_channel=True)
    for key in spec.single_matrix:
        if out.get(key) is not None:
            # [..., N, R, C]: the token dim sits before two trailing dims.
            out[key] = F.pad(out[key], [0, 0, 0, 0, 0, pad])
    return out


def pad_trunk_tokens(
    tensors: dict[str, torch.Tensor],
    n_true: int,
    spec: TrunkPadSpec,
    feature_dict: dict | None = None,
    *,
    align: int = TOKEN_ALIGN,
) -> tuple[dict[str, torch.Tensor], dict | None, int]:
    """Zero-pad every tensor named in ``spec`` up to a multiple of ``align``.

    ``tensors`` maps the same names used in ``spec`` to their current
    tensors. Returns ``(padded_tensors, padded_feature_dict, n_true)``;
    ``n_true`` is handed back unchanged so the caller can slice back down to
    it after running its layer stack. A no-op (same objects back) when
    ``n_true`` is already aligned.
    """
    n_pad = round_up_tokens(n_true, align)
    if n_pad == n_true:
        return tensors, feature_dict, n_true
    pad = n_pad - n_true
    out = dict(tensors)
    for key in spec.single_channel:
        out[key] = pad_single_token_dim(out[key], pad, has_channel=True)
    for key in spec.pair_channel:
        out[key] = pad_pair_token_dims(out[key], pad, has_channel=True)
    for key in spec.single_last:
        out[key] = pad_single_token_dim(out[key], pad, has_channel=False)
    for key in spec.pair_last:
        out[key] = pad_pair_token_dims(out[key], pad, has_channel=False)
    padded_features = (
        pad_feature_dict(feature_dict, pad, spec.feature_dict)
        if feature_dict is not None and spec.feature_dict is not None
        else feature_dict
    )
    return out, padded_features, n_true


# Token dims of each unpad kind, counted from the end. "single" and "pair"
# alias the channel kinds for the [B, N, C] s / [B, N, N, C] z tensors.
_TOKEN_DIMS = {
    "single_last": (-1,),
    "single_channel": (-2,),
    "single": (-2,),
    "pair_last": (-2, -1),
    "pair_channel": (-3, -2),
    "pair": (-3, -2),
}


def unpad_trunk_tokens(
    *tensors: torch.Tensor, n_true: int, kinds: tuple[str, ...] | None = None
) -> tuple[torch.Tensor, ...]:
    """Slice each tensor's token dim(s) back down to ``n_true``, if padded.

    ``kinds`` names each tensor's shape explicitly with a ``TrunkPadSpec``
    field name -- "single_channel" (``[..., N, C]``), "single_last"
    (``[..., N]``), "pair_channel" (``[..., N, N, C]``) or "pair_last"
    (``[..., N, N]``) -- matching ``pad_trunk_tokens``'s own spec-driven,
    non-heuristic approach; "single" and "pair" alias the two channel kinds.
    When omitted, falls back to ``ndim`` (3 -> single, 4 -> pair), which is
    reliable for the plain ``s``/``z`` tensors this is usually called with
    but not for other shapes; pass ``kinds`` explicitly for anything else.
    A tensor whose token dims already equal ``n_true`` comes back as is.
    """
    if kinds is not None and len(kinds) != len(tensors):
        raise ValueError(f"kinds has {len(kinds)} entries for {len(tensors)} tensors")
    out = []
    for i, t in enumerate(tensors):
        kind = kinds[i] if kinds is not None else ("single" if t.ndim == 3 else "pair")
        if kind not in _TOKEN_DIMS:
            raise ValueError(f"unknown token-padding kind {kind!r}")
        token_dims = _TOKEN_DIMS[kind]
        if all(t.shape[d] == n_true for d in token_dims):
            out.append(t)
            continue
        index = [slice(None)] * t.ndim
        for d in token_dims:
            index[d] = slice(n_true)
        out.append(t[tuple(index)])
    return tuple(out)
