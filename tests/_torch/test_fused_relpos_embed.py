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

"""Tests for the fused relative-position embedding Triton kernel.

Covers correctness of the three call-site variants (OF3, Boltz-2, Protenix)
and the chunked-row path, verified against the existing PyTorch references
with TF32 disabled so the comparison is exact float32.
"""

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.layers.position_encoders import RelativePositionEncoder
from bionemo_ir._torch.modules.openfold3.utils.relpos import relpos_complex
from bionemo_ir._torch.utils import dist_one_hot
from bionemo_ir.dsl_kernels.triton.fused_relpos_embed import fused_relpos_embed

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_batch(
    batch_shape: tuple[int, ...],
    n_tokens: int,
    n_chains: int,
    *,
    dtype: torch.dtype = torch.int32,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Synthetic batch with heterogeneous chains, multi-token residues, and two entities."""
    chain_len = n_tokens // n_chains
    tok = torch.arange(n_tokens, device=device)
    asym_id = (tok // chain_len).clamp_max(n_chains - 1)
    residue_index = tok - asym_id * chain_len
    # Make token 1 share residue 0 on the same chain (multi-token residue)
    if n_tokens > 2:
        residue_index[1] = residue_index[0]
    sym_id = asym_id.clone()
    entity_id = asym_id % 2  # 2 entities: chains alternate entity 0/1

    def expand(*shape):
        return lambda t: t.expand(*shape, -1).to(dtype=dtype, device=device).contiguous()

    ex = expand(*batch_shape)
    return {
        "residue_index": ex(residue_index),
        "token_index": ex(tok),
        "sym_id": ex(sym_id),
        "asym_id": ex(asym_id),
        "entity_id": ex(entity_id),
    }


# ---------------------------------------------------------------------------
# OF3 path: fused_relpos_embed vs relpos_complex + F.linear
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("n_tokens", [1, 7, 64, 128])
@pytest.mark.parametrize("n_chains", [1, 2])
@pytest.mark.parametrize("max_relative_idx,max_relative_chain", [(2, 1), (32, 2)])
def test_of3_matches_relpos_complex(
    n_tokens: int, n_chains: int, max_relative_idx: int, max_relative_chain: int
) -> None:
    if n_tokens < n_chains:
        pytest.skip("fewer tokens than chains")
    B = 2
    device = "cuda"
    batch = _make_batch((B,), n_tokens, n_chains, device=device)

    n_rel = 2 * max_relative_idx + 2
    n_chain = 2 * max_relative_chain + 2
    C_in = 2 * n_rel + 1 + n_chain
    C_out = 32
    weight = torch.randn(C_out, C_in, device=device)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = F.linear(relpos_complex(batch, max_relative_idx, max_relative_chain), weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    z = torch.zeros(B, n_tokens, n_tokens, C_out, device=device)
    got = fused_relpos_embed(
        z,
        batch["residue_index"],
        batch["token_index"],
        batch["sym_id"],
        batch["asym_id"],
        batch["entity_id"],
        weight,
        max_relative_idx=max_relative_idx,
        max_relative_chain=max_relative_chain,
        entity_chain_cond=True,
    )

    assert torch.allclose(ref, got, atol=1e-5, rtol=0), f"max error {(ref - got).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Chunked path: row-sliced output matches full output
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("n_tokens,chunk", [(16, 4), (33, 8), (64, 16)])
def test_chunked_rows_match_full(n_tokens: int, chunk: int) -> None:
    B, C_out = 1, 32
    device = "cuda"
    batch = _make_batch((B,), n_tokens, 2, device=device)
    weight = torch.randn(C_out, 139, device=device)

    def _embed(z, *, row_start=0):
        return fused_relpos_embed(
            z,
            batch["residue_index"],
            batch["token_index"],
            batch["sym_id"],
            batch["asym_id"],
            batch["entity_id"],
            weight,
            entity_chain_cond=True,
            row_start=row_start,
        )

    # Full reference
    z_full = torch.zeros(B, n_tokens, n_tokens, C_out, device=device)
    _embed(z_full)

    # Chunked: write each block into a pre-allocated output
    z_chunked = torch.zeros_like(z_full)
    for start in range(0, n_tokens, chunk):
        length = min(chunk, n_tokens - start)
        z_rows = torch.zeros(B, length, n_tokens, C_out, device=device)
        _embed(z_rows, row_start=start)
        z_chunked[:, start : start + length].copy_(z_rows)

    assert torch.equal(z_full, z_chunked)


# ---------------------------------------------------------------------------
# RelativePositionEncoder fused path: Boltz-2 and Protenix
# ---------------------------------------------------------------------------


def _encoder(device: str, **kwargs) -> RelativePositionEncoder:
    """A ``RelativePositionEncoder`` with seeded weights (``Linear`` leaves them uninitialized)."""
    enc = RelativePositionEncoder(**kwargs).to(device)
    with torch.no_grad():
        enc.linear.weight.normal_(generator=torch.Generator(device=device).manual_seed(0))
    return enc


@requires_cuda
@pytest.mark.parametrize("fix_sym_check", [False, True], ids=["boltz2", "protenix"])
@pytest.mark.parametrize("n_tokens", [7, 64])
def test_relative_position_encoder_fused_matches_pytorch(fix_sym_check: bool, n_tokens: int) -> None:
    """RelativePositionEncoder.forward() fused path matches the F.embedding reference."""
    B, C_out = 2, 64
    device = "cuda"
    enc = _encoder(device, token_z=C_out, r_max=32, s_max=2, fix_sym_check=fix_sym_check)
    batch = _make_batch((B,), n_tokens, 2, device=device)

    # Reference: force PyTorch F.embedding path (no fused kernel)
    d_res, d_tok, d_chain, same_ent = enc._relp_buckets(
        batch["asym_id"],
        batch["residue_index"],
        batch["entity_id"],
        batch["token_index"],
        batch["sym_id"],
    )
    n_pos = 2 * enc.r_max + 2
    n_chain = 2 * enc.s_max + 2
    wt = enc.linear.weight.t()
    ref = (
        F.embedding(d_res, wt[:n_pos])
        + F.embedding(d_tok, wt[n_pos : 2 * n_pos])
        + same_ent[..., None].float() * wt[2 * n_pos]
        + F.embedding(d_chain, wt[2 * n_pos + 1 : 2 * n_pos + 1 + n_chain])
    )

    # Fused path via forward()
    got = enc(
        asym_id=batch["asym_id"],
        residue_index=batch["residue_index"],
        entity_id=batch["entity_id"],
        token_index=batch["token_index"],
        sym_id=batch["sym_id"],
    )

    assert torch.allclose(ref, got, atol=1e-5, rtol=0), (
        f"fix_sym_check={fix_sym_check} max error {(ref - got).abs().max().item():.2e}"
    )


@requires_cuda
def test_relative_position_encoder_precomputed_relp_unchanged() -> None:
    """When relp is precomputed, RelativePositionEncoder bypasses the fused kernel."""
    B, N, C_out = 1, 32, 64
    device = "cuda"
    enc = _encoder(device, token_z=C_out, r_max=32, s_max=2)
    batch = _make_batch((B,), N, 2, device=device)

    # generate_relp requires int64 for F.one_hot
    batch64 = {k: v.to(torch.int64) for k, v in batch.items()}
    relp = enc.generate_relp(
        asym_id=batch64["asym_id"],
        residue_index=batch64["residue_index"],
        entity_id=batch64["entity_id"],
        token_index=batch64["token_index"],
        sym_id=batch64["sym_id"],
    )
    ref = enc.linear(relp)
    got = enc(relp=relp)
    assert torch.equal(ref, got)


# ---------------------------------------------------------------------------
# Accumulation: kernel adds into existing z, does not overwrite
# ---------------------------------------------------------------------------


@requires_cuda
def test_kernel_accumulates_into_existing_z() -> None:
    B, N, C_out = 1, 16, 32
    device = "cuda"
    batch = _make_batch((B,), N, 1, device=device)
    weight = torch.randn(C_out, 139, device=device)

    z_ones = torch.ones(B, N, N, C_out, device=device)

    def _embed(z):
        return fused_relpos_embed(
            z,
            batch["residue_index"],
            batch["token_index"],
            batch["sym_id"],
            batch["asym_id"],
            batch["entity_id"],
            weight,
            entity_chain_cond=True,
        )

    got = _embed(z_ones.clone())

    z_zero = torch.zeros(B, N, N, C_out, device=device)
    expected = _embed(z_zero) + 1.0

    assert torch.allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# OF2 multimer path: has_token_feat=False
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("n_tokens,n_chains", [(64, 2), (256, 4)])
def test_of2_multimer_matches_reference(n_tokens: int, n_chains: int) -> None:
    """has_token_feat=False matches the OF2 multimer relpos reference (no rel_token column)."""
    B = 2
    device = "cuda"
    batch = _make_batch((B,), n_tokens, n_chains, device=device)

    max_relative_idx, max_relative_chain = 32, 2
    n_rel = 2 * max_relative_idx + 2
    n_chain = 2 * max_relative_chain + 2
    C_in = n_rel + 1 + n_chain  # 73 — no rel_token block
    C_out = 64
    weight = torch.randn(C_out, C_in, device=device)
    bias = torch.randn(C_out, device=device)

    # Reference: OF2 multimer formula with dist_one_hot
    asym_id = batch["asym_id"]
    res_idx = batch["residue_index"]
    sym_id = batch["sym_id"]
    entity_id = batch["entity_id"]
    rel_pos_b = torch.arange(0, n_rel, device=device)
    rel_chain_b = torch.arange(0, n_chain, device=device)
    asym_same = asym_id[:, :, None] == asym_id[:, None, :]
    ent_same = entity_id[:, :, None] == entity_id[:, None, :]
    offset = res_idx[:, :, None] - res_idx[:, None, :]
    clipped = torch.clamp(offset + max_relative_idx, 0, 2 * max_relative_idx)
    final_pos = torch.where(asym_same, clipped, torch.full_like(clipped, 2 * max_relative_idx + 1))
    rel_pos = dist_one_hot(final_pos, rel_pos_b)
    sym_diff = sym_id[:, :, None] - sym_id[:, None, :]
    clipped_chain = torch.clamp(sym_diff + max_relative_chain, 0, 2 * max_relative_chain)
    final_chain = torch.where(ent_same, clipped_chain, torch.full_like(clipped_chain, 2 * max_relative_chain + 1))
    rel_chain = dist_one_hot(final_chain, rel_chain_b)
    rel_feat = torch.cat([rel_pos, ent_same[..., None].float(), rel_chain], dim=-1)
    ref = F.linear(rel_feat, weight, bias)

    # Fused kernel
    z = torch.zeros(B, n_tokens, n_tokens, C_out, device=device)
    fused_relpos_embed(
        z,
        res_idx,
        None,
        sym_id,
        asym_id,
        entity_id,
        weight,
        max_relative_idx=max_relative_idx,
        max_relative_chain=max_relative_chain,
        entity_chain_cond=True,
        has_token_feat=False,
    )
    z = z + bias

    assert torch.allclose(ref, z, atol=1e-5, rtol=0), f"max error {(ref - z).abs().max().item():.2e}"
