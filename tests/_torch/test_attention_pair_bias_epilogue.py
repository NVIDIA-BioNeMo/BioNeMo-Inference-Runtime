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
"""Output-gated pair-bias attention with a residual: the fused epilogue against the unfused path."""

import json
import math

import pytest
import torch

from bionemo_ir._torch.custom_ops.attn_epilogue import SLAB, AttnEpilogue
from bionemo_ir._torch.custom_ops.attn_epilogue._config import tuned_configs
from bionemo_ir._torch.custom_ops.attn_epilogue.cutedsl import AttnEpilogueCuTe
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.transformers.diffusion_transformer import ProtenixDiffusionTransformer
from bionemo_ir.configs import DiffusionTransformerConfig
from tests._torch import cutedsl_test_modes, skip_if_epilogue_tile_exceeds_smem

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# The atom transformers' shape, the one the resident epilogue kernel serves.
CHANNELS, PAIR_CHANNELS, HEADS = 128, 16, 4
N_QUERIES, N_KEYS = 32, 128
# The diffusion token transformers' shape, the one the streamed SM90 kernel serves.
TOKEN_CHANNELS, TOKEN_PAIR_CHANNELS, TOKEN_HEADS, TOKEN_COND = 768, 128, 16, 384


def _init(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim == 2:
                parameter.normal_(0, parameter.shape[1] ** -0.5)
            else:
                parameter.normal_(1 if name.endswith("weight") else 0, 0.1)


def _epilogues(module: torch.nn.Module) -> list[AttentionPairBias]:
    attentions = [child for child in module.modules() if isinstance(child, AttentionPairBias)]
    if not attentions or any(attention._epilogue is None for attention in attentions):
        pytest.skip("the fused epilogue needs SM80+ and the CuTeDSL sources or CUBINs")
    return attentions


def _count_served(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    served: list[bool] = []
    call = AttnEpilogue.__call__

    def counting(self, *args, **kwargs):
        out = call(self, *args, **kwargs)
        served.append(out is not None)
        return out

    monkeypatch.setattr(AttnEpilogue, "__call__", counting)
    return served


def _unfused(attentions: list[AttentionPairBias], run):
    epilogues = [attention._epilogue for attention in attentions]
    for attention in attentions:
        attention._epilogue = None
    try:
        return run()
    finally:
        for attention, epilogue in zip(attentions, epilogues, strict=True):
            attention._epilogue = epilogue


@pytest.mark.parametrize("samples", [1, 5])
def test_token_rows_share_one_gate_row_across_samples(monkeypatch: pytest.MonkeyPatch, samples: int) -> None:
    torch.manual_seed(0)
    attention = AttentionPairBias(
        layer_idx=0,
        c_s=CHANNELS,
        c_z=PAIR_CHANNELS,
        num_heads=HEADS,
        initial_norm=False,
        bias_proj=True,
        output_gate_dim=CHANNELS,
        dtype=torch.bfloat16,
        attn_backend="SDPA",
    ).cuda()
    _init(attention)
    attentions = _epilogues(attention)
    n = 96
    s = torch.randn(1, samples, n, CHANNELS, device="cuda", dtype=torch.bfloat16)
    # The conditioning holds one row per token, broadcast over the samples.
    cond = torch.randn(1, 1, n, CHANNELS, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(1, n, n, PAIR_CHANNELS, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, n, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(s)
    served = _count_served(monkeypatch)
    with torch.inference_mode():

        def run():
            return attention(s, z, mask, single_embedding=cond, residual=residual)

        expected = _unfused(attentions, run)
        actual = run()

    assert served == [True]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_windowed_atom_attention_projects_keys_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    config = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=HEADS,
        dim=CHANNELS,
        dim_single_cond=CHANNELS,
        dim_pairwise=PAIR_CHANNELS,
        dtype="bfloat16",
    )
    model = ProtenixDiffusionTransformer(config).cuda().eval()
    _init(model)
    attentions = _epilogues(model)
    batch, atoms = 3, 200
    blocks = math.ceil(atoms / N_QUERIES)
    q = torch.randn(batch, atoms, CHANNELS, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(batch, atoms, CHANNELS, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(batch, blocks, N_QUERIES, N_KEYS, PAIR_CHANNELS, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, atoms, device="cuda", dtype=torch.bfloat16)
    metadata = model.build_attn_metadata(blocks, N_QUERIES, N_KEYS, q.device)
    served = _count_served(monkeypatch)
    with torch.inference_mode():

        def run():
            return model(q, c, p, mask, N_QUERIES, N_KEYS, metadata)

        expected = _unfused(attentions, run)
        actual = run()

    assert served == [True] * config.num_blocks
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", ["CuTeDSL", "SDPA"])
@pytest.mark.parametrize(
    ("tokens", "samples"),
    # Rows nearest the channel-tiled SM80 anchors, then nearest a streamed SM90 one.
    [(96, 1), (200, 5), (300, 5)],
    ids=["short_aligned", "short_ragged_samples", "long_ragged_samples"],
)
def test_token_width_takes_the_streamed_kernel(
    monkeypatch: pytest.MonkeyPatch, backend: str, tokens: int, samples: int
) -> None:
    torch.manual_seed(0)
    attention = AttentionPairBias(
        layer_idx=0,
        c_s=TOKEN_CHANNELS,
        c_z=TOKEN_PAIR_CHANNELS,
        num_heads=TOKEN_HEADS,
        initial_norm=False,
        bias_proj=True,
        output_gate_dim=TOKEN_COND,
        dtype=torch.bfloat16,
        attn_backend=backend,
    ).cuda()
    _init(attention)
    attentions = _epilogues(attention)
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    s = torch.randn(1, samples, tokens, TOKEN_CHANNELS, **bf16)
    cond = torch.randn(1, 1, tokens, TOKEN_COND, **bf16)
    z = torch.randn(1, tokens, tokens, TOKEN_PAIR_CHANNELS, **bf16)
    mask = torch.ones(1, tokens, **bf16)
    residual = torch.randn_like(s)
    served = _count_served(monkeypatch)
    with torch.inference_mode():

        def run():
            return attention(s, z, mask, single_embedding=cond, residual=residual)

        expected = _unfused(attentions, run)
        actual = run()

    assert served == [True]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_token_transformer_layers_take_the_streamed_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    config = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=TOKEN_HEADS,
        dim=TOKEN_CHANNELS,
        dim_single_cond=TOKEN_COND,
        dim_pairwise=TOKEN_PAIR_CHANNELS,
        bias_proj=True,
        dtype="bfloat16",
        pairwise_attention_backend="CuTeDSL",
    )
    model = ProtenixDiffusionTransformer(config).cuda().eval()
    _init(model)
    attentions = _epilogues(model)
    samples, tokens = 3, 150
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    a = torch.randn(samples, tokens, TOKEN_CHANNELS, **bf16)
    s = torch.randn(samples, tokens, TOKEN_COND, **bf16)
    z = torch.randn(1, tokens, tokens, TOKEN_PAIR_CHANNELS, **bf16)
    mask = torch.ones(1, tokens, **bf16)
    served = _count_served(monkeypatch)
    with torch.inference_mode():

        def run():
            return model(a, s, z, mask)

        expected = _unfused(attentions, run)
        actual = run()

    assert served == [True] * config.num_blocks
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


def _sm8x_token_width_tiles() -> list[tuple[int, int]]:
    """``(sm, anchor)`` once for every distinct tile the SM8x token-width tunings ship, lowest SM first."""
    cases, seen = [], set()
    for sm in (80, 86, 89):
        for tuning in tuned_configs(sm, TOKEN_CHANNELS // SLAB, SLAB, TOKEN_CHANNELS):
            tile = (tuning.kernel_abi, tuning.kernel_variant, json.dumps(tuning.tile_params, sort_keys=True))
            if tile not in seen:
                seen.add(tile)
                cases.append((sm, tuning.rows))
    return cases


@pytest.mark.parametrize(("sm", "anchor"), _sm8x_token_width_tiles(), ids=lambda value: str(value))
def test_sm8x_token_width_tunings_add_the_gated_update(sm: int, anchor: int) -> None:
    """Each SM8x token-width tile, built from source for this GPU, adds the output-gated projection.

    The layer's 16 heads of 48 channels reach the kernel as 12 slabs of 64, and 150 tokens leave a
    partial J tile for both tile heights.
    """
    if "source" not in cutedsl_test_modes("bionemo_ir._torch.custom_ops.attn_epilogue._source"):
        pytest.skip("SM80 CUBINs do not run on this device; only the source path can target it")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the SM80 kernel needs an SM80 or newer GPU")
    heads, head_dim = TOKEN_CHANNELS // SLAB, SLAB
    backend = AttnEpilogueCuTe(heads, head_dim, TOKEN_CHANNELS, has_bias=True, has_output_gate=True, anchor=anchor)
    backend._sm_version = sm
    skip_if_epilogue_tile_exceeds_smem(backend)
    op = AttnEpilogue(
        backend,
        heads,
        head_dim,
        TOKEN_CHANNELS,
        has_bias=True,
        has_output_gate=True,
        layer_heads=TOKEN_HEADS,
        layer_head_dim=TOKEN_CHANNELS // TOKEN_HEADS,
    )
    torch.manual_seed(anchor)
    samples, tokens = 3, 150
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    mha_o = torch.randn(samples, tokens, TOKEN_HEADS, TOKEN_CHANNELS // TOKEN_HEADS, **bf16)
    gate = torch.randn(1, samples, tokens, TOKEN_CHANNELS, **bf16)
    weight = (torch.randn(TOKEN_CHANNELS, TOKEN_CHANNELS, device="cuda") * TOKEN_CHANNELS**-0.5).to(torch.bfloat16)
    bias = (torch.randn(TOKEN_CHANNELS, device="cuda") * 0.1).to(torch.bfloat16)
    residual = torch.randn(1, samples, tokens, TOKEN_CHANNELS, **bf16)
    # One gate row per token, broadcast over the samples, as the conditioning is.
    logits = (2 * torch.randn(1, 1, tokens, TOKEN_CHANNELS, device="cuda")).to(torch.bfloat16)
    update = torch.nn.functional.linear(mha_o.reshape(gate.shape) * gate.sigmoid(), weight, bias)
    # The gated update joins the residual add, which rounds once.
    expected = (residual.float() + logits.sigmoid().float() * update.float()).to(torch.bfloat16)

    with torch.inference_mode():
        actual = op(mha_o, gate, weight, residual, bias=bias, output_gate=logits)

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_unfused_output_takes_the_precomputed_gate_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the fused epilogue, precomputed output-gate logits replace the gate projection even when
    ``single_embedding`` is given (the token layers pass both)."""
    torch.manual_seed(0)
    attention = AttentionPairBias(
        layer_idx=0,
        c_s=CHANNELS,
        c_z=PAIR_CHANNELS,
        num_heads=HEADS,
        initial_norm=False,
        bias_proj=True,
        output_gate_dim=CHANNELS,
        dtype=torch.bfloat16,
        attn_backend="SDPA",
    ).cuda()
    _init(attention)
    attention._epilogue = None
    n = 64
    s = torch.randn(1, 1, n, CHANNELS, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(1, 1, n, CHANNELS, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(1, n, n, PAIR_CHANNELS, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, n, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(s)
    with torch.inference_mode():
        logits = attention.output_projection(cond)
        expected = attention(s, z, mask, single_embedding=cond, residual=residual)
        monkeypatch.setattr(attention, "_output_gate_op", lambda *args, **kwargs: pytest.fail("projected again"))
        actual = attention(s, z, mask, single_embedding=cond, residual=residual, output_gate_logits=logits)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
