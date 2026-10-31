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
"""dual_norm_qk against the fused norms it replaces, and the attention layer and conditioned block that call it."""

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.layers import normalization
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.normalization import (
    FusedLayerNorm,
    FusedRMSNorm,
    replace_with_fused_layernorm,
    replace_with_fused_rmsnorm,
)
from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock
from bionemo_ir.dsl_kernels.triton.dual_norm_qk import dual_norm_qk
from tests._torch import init_module_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _norms(kind: str, width: int, dtype: torch.dtype, affine: bool, seed: int) -> tuple[nn.Module, nn.Module]:
    g = torch.Generator().manual_seed(seed)
    pair = []
    for _ in range(2):
        if kind == "rms":
            norm = FusedRMSNorm.from_rmsnorm(nn.RMSNorm(width, eps=1e-5, elementwise_affine=affine, dtype=dtype))
        else:
            norm = FusedLayerNorm.from_layernorm(
                nn.LayerNorm(width, eps=1e-5, elementwise_affine=affine, bias=affine, dtype=dtype)
            )
        with torch.no_grad():
            for p in norm.parameters():
                p.copy_(
                    1 + 0.2 * torch.randn(p.shape, generator=g)
                    if p is norm.weight
                    else 0.2 * torch.randn(p.shape, generator=g)
                )
        pair.append(norm.cuda())
    return pair[0], pair[1]


def _call(
    q_norm: nn.Module, k_norm: nn.Module, q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    rms = isinstance(q_norm, FusedRMSNorm)
    return dual_norm_qk(
        q,
        k,
        q_norm.weight,
        k_norm.weight,
        None if rms else q_norm.bias,
        None if rms else k_norm.bias,
        eps=1e-5,
        rms_norm=rms,
    )


@pytest.mark.parametrize("kind", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [768, 384, 256, 192, 64])
@pytest.mark.parametrize("lead", [(2, 400), (1, 208), (3, 7), (1, 1)])
@pytest.mark.parametrize("affine", [True, False])
def test_matches_the_fused_norms_on_slices_of_a_projection(
    kind: str, dtype: torch.dtype, width: int, lead: tuple[int, ...], affine: bool
) -> None:
    """q and k as column slices of a [*, 4 width] projection, as an attention layer splits it: each output is the
    module's on the contiguous copy, bit for bit, and contiguous."""
    q_norm, k_norm = _norms(kind, width, dtype, affine, seed=width)
    projected = torch.randn(*lead, 4 * width, device="cuda").to(dtype)
    q, _, k, _ = projected.split(width, dim=-1)
    out = _call(q_norm, k_norm, q, k)
    assert out is not None
    q_out, k_out = out
    assert q_out.is_contiguous() and k_out.is_contiguous() and q_out.shape == q.shape and k_out.shape == k.shape
    assert torch.equal(q_out, q_norm(q)) and torch.equal(k_out, k_norm(k))


def test_declines_what_one_launch_cannot_serve() -> None:
    """Different row counts or widths, unevenly spaced rows and CPU tensors fall back to the modules."""
    q_norm, k_norm = _norms("rms", 64, torch.bfloat16, True, seed=1)
    x = torch.randn(2, 9, 256, device="cuda").to(torch.bfloat16)
    q, k = x[..., :64], x[..., 64:128]
    assert _call(q_norm, k_norm, q, k[:, :5]) is None  # rows differ
    assert _call(q_norm, k_norm, q, x[..., 64:192]) is None  # widths differ
    assert _call(q_norm, k_norm, q[:, ::2], k[:, ::2]) is None  # 5 of 9 rows: the batch step breaks the spacing
    assert _call(q_norm, k_norm, q.transpose(0, 1), k.transpose(0, 1)) is None  # rows not evenly spaced
    assert _call(q_norm, k_norm, q.cpu(), k.cpu()) is None


def test_reads_evenly_spaced_rows_in_place() -> None:
    """Every other row of a 10-row batch keeps one stride across the batch boundary, so it needs no copy."""
    q_norm, k_norm = _norms("rms", 64, torch.bfloat16, True, seed=1)
    x = torch.randn(2, 10, 256, device="cuda").to(torch.bfloat16)
    q, k = x[:, ::2, :64], x[:, ::2, 64:128]
    out = _call(q_norm, k_norm, q, k)
    assert out is not None
    assert torch.equal(out[0], q_norm(q)) and torch.equal(out[1], k_norm(k))


@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
def test_attention_with_qk_norm_runs_one_launch_and_matches_the_modules(
    norm_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AttentionPairBias normalizes q and k through dual_norm_qk once its norms are the fused ones, with the
    modules' output bit for bit."""
    torch.manual_seed(0)
    layer = AttentionPairBias(
        layer_idx=0,
        c_s=768,
        c_z=128,
        num_heads=12,
        initial_norm=True,
        norm_type=norm_type,
        use_qk_norm=True,
        dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        # Linear allocates its parameters uninitialized.
        layer.in_proj.weight.normal_(0, 768**-0.5)
        layer.in_proj.bias.normal_(0, 0.1)
        for name in ("q_norm", "k_norm"):
            norm = getattr(layer, name)
            fused = FusedRMSNorm.from_rmsnorm(norm) if norm_type == "rms_norm" else FusedLayerNorm.from_layernorm(norm)
            fused.weight.uniform_(0.5, 1.5)
            if getattr(fused, "bias", None) is not None:
                fused.bias.normal_(0, 0.2)
            setattr(layer, name, fused)
    s = torch.randn(2, 37, 768, device="cuda").to(torch.bfloat16)
    calls = []
    real = layer._norm_qk.__func__

    def counting(self: AttentionPairBias, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(1)
        return real(self, q, k)

    monkeypatch.setattr(AttentionPairBias, "_norm_qk", counting)
    with torch.inference_mode():
        q, k, _, _ = layer._prep_qkvg(s, s)
        projected = layer.in_proj(s)
        q_ref, _, k_ref, _ = projected.split(layer._in_proj_sizes, dim=-1)
        assert torch.equal(q, layer.q_norm(q_ref)) and torch.equal(k, layer.k_norm(k_ref))
    assert calls == [1]


@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
def test_conditioned_block_normalizes_its_condition_once_for_both_adalns(
    norm_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attention and transition AdaLNs normalize the block's one condition in a single launch, and the block's
    output stays bit for bit what each AdaLN projecting on its own gives."""
    torch.manual_seed(3)
    block = ConditionedTransformerBlock(
        0, 128, 16, 64, 2, 2, "SDPA", 1e6, False, 1e-5, torch.bfloat16, norm_type=norm_type
    ).cuda()
    init_module_weights(block)
    replace_with_fused_layernorm(block)
    replace_with_fused_rmsnorm(block)
    with torch.no_grad():
        # init_module_weights sets every norm scale to 1; distinct gammas show each AdaLN gets its own.
        for adaln in (block.adaln, block.transition.adaln):
            adaln.s_norm.weight.uniform_(0.5, 1.5)
    mask = torch.arange(12, device="cuda") < torch.tensor([[12], [9]], device="cuda")
    pair_mask = mask[:, :, None] & mask[:, None, :]
    x = torch.randn(2, 12, 128, device="cuda").to(torch.bfloat16).masked_fill(~mask[..., None], 0)
    pair = torch.randn(2, 12, 12, 16, device="cuda").to(torch.bfloat16).masked_fill(~pair_mask[..., None], 0)
    condition = torch.randn(2, 12, 64, device="cuda").to(torch.bfloat16)

    widths = []
    real = normalization.dual_norm_qk

    def counting(*args: torch.Tensor, **kwargs: float | bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        widths.append(args[0].shape[-1])
        return real(*args, **kwargs)

    monkeypatch.setattr(normalization, "dual_norm_qk", counting)
    with torch.inference_mode():
        actual = block(x, pair, condition, mask)
        sanitized = condition.masked_fill(~mask[..., None], 0)
        separate = (block.adaln.project_condition(sanitized), block.transition.adaln.project_condition(sanitized))
        expected = block(x, pair, condition, mask, projected_condition=separate)
    assert widths == [64]
    assert torch.equal(actual, expected)
