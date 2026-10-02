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

"""Fused relative-position embedding kernel.

Replaces the three-step baseline::

    scatter_(-1, bin, 1.0)  ×N → [B, N, N, C_in] one-hot intermediates
    torch.cat               → [B, N, N, C_in]
    F.linear / F.embedding  → [B, N, N, C_out]

with a single Triton kernel that computes bin indices per (b,i,j),
looks up the corresponding weight rows, and accumulates directly into the
output tensor.  The [B, N, N, C_in] intermediate is never materialised.

Two C_in layouts are supported via ``has_token_feat``:

    has_token_feat=True  (OF3 / Protenix / Boltz-2):
        [ rel_pos (n_rel) | rel_token (n_rel) | same_entity (1) | rel_chain (n_chain) ]
        C_in = 2*n_rel + 1 + n_chain

    has_token_feat=False  (OF2 multimer):
        [ rel_pos (n_rel) | same_entity (1) | rel_chain (n_chain) ]
        C_in = n_rel + 1 + n_chain

where ``n_rel = 2*max_relative_idx + 2`` and ``n_chain = 2*max_relative_chain + 2``.

The chain-bin condition also differs across models:

* OF3 / Protenix / OF2 multimer (``entity_chain_cond=True``):
    chain offset valid when ``same_entity``; out-of-range bin otherwise.
* Boltz-2 (``entity_chain_cond=False``):
    chain offset valid when ``~same_chain``; out-of-range bin otherwise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "N_rows", "row_start"])
def _fused_relpos_embed_kernel(
    res_ptr,
    tok_ptr,  # [B, N] int32 — unused when has_token_feat=False
    sym_ptr,
    asym_ptr,
    entity_ptr,  # [B, N] int32
    wT_ptr,  # [C_in, C_out] float32, contiguous (weight transposed)
    z_ptr,  # [B, N_rows, N, C_out] float32, accumulate into existing values
    N,  # total token count (j dimension)
    N_rows,  # row count in z (may be < N for chunked encoding)
    row_start,  # global row offset: global_i = local_i + row_start
    C_out: tl.constexpr,
    n_rel: tl.constexpr,  # 2*max_relative_idx + 2
    rel_clip: tl.constexpr,  # max_relative_idx
    n_chain: tl.constexpr,  # 2*max_relative_chain + 2
    chain_clip: tl.constexpr,  # max_relative_chain
    entity_chain_cond: tl.constexpr,  # True=same_entity gate, False=same_chain gate
    has_token_feat: tl.constexpr,  # True=OF3/Protenix/Boltz-2, False=OF2 multimer
    BLOCK_J: tl.constexpr,
):
    """One program handles one (b, i_local) row, iterating over all j."""
    bi = tl.program_id(0).to(tl.int64)
    b = bi // N_rows
    i_local = bi % N_rows
    i = i_local + row_start  # global token index for feature lookup

    c = tl.arange(0, C_out)

    base_i = b * N + i
    res_i = tl.load(res_ptr + base_i).to(tl.int32)
    asym_i = tl.load(asym_ptr + base_i).to(tl.int32)
    ent_i = tl.load(entity_ptr + base_i).to(tl.int32)
    if has_token_feat:
        sym_i = tl.load(sym_ptr + base_i).to(tl.int32)
        tok_i = tl.load(tok_ptr + base_i).to(tl.int32)
    else:
        sym_i = tl.load(sym_ptr + base_i).to(tl.int32)

    # Entity weight row: bin index = 2*n_rel (has_token_feat) or n_rel (no token)
    if has_token_feat:
        w_ent_row = tl.load(wT_ptr + (2 * n_rel) * C_out + c)
    else:
        w_ent_row = tl.load(wT_ptr + n_rel * C_out + c)

    for j0 in range(0, N, BLOCK_J):
        j_offs = j0 + tl.arange(0, BLOCK_J)
        mask = j_offs < N

        bj = b * N + j_offs
        res_j = tl.load(res_ptr + bj, mask, res_i).to(tl.int32)
        sym_j = tl.load(sym_ptr + bj, mask, sym_i).to(tl.int32)
        asym_j = tl.load(asym_ptr + bj, mask, asym_i).to(tl.int32)
        ent_j = tl.load(entity_ptr + bj, mask, ent_i).to(tl.int32)

        same_chain = asym_i == asym_j
        same_ent = ent_i == ent_j

        # rel_pos: residue offset, clipped to [0, 2*rel_clip], out-bin when cross-chain
        off_pos = tl.minimum(tl.maximum(res_i - res_j + rel_clip, 0), 2 * rel_clip)
        bin_pos = tl.where(same_chain, off_pos, 2 * rel_clip + 1)

        w_pos = tl.load(wT_ptr + bin_pos[:, None] * C_out + c[None, :])

        if has_token_feat:
            tok_j = tl.load(tok_ptr + bj, mask, tok_i).to(tl.int32)
            same_res = same_chain & (res_i == res_j)
            off_tok = tl.minimum(tl.maximum(tok_i - tok_j + rel_clip, 0), 2 * rel_clip)
            bin_tok = tl.where(same_res, off_tok, 2 * rel_clip + 1) + n_rel
            w_tok = tl.load(wT_ptr + bin_tok[:, None] * C_out + c[None, :])

        # rel_chain: sym-id offset, condition depends on model variant
        off_chain = tl.minimum(tl.maximum(sym_i - sym_j + chain_clip, 0), 2 * chain_clip)
        if entity_chain_cond:
            # OF3 / Protenix / OF2 multimer: valid when same entity
            bin_chain = tl.where(same_ent, off_chain, 2 * chain_clip + 1)
        else:
            # Boltz-2: valid when different chain (cross-chain symmetry)
            bin_chain = tl.where(same_chain, 2 * chain_clip + 1, off_chain)
        if has_token_feat:
            bin_chain = bin_chain + 2 * n_rel + 1
        else:
            bin_chain = bin_chain + n_rel + 1

        w_chain = tl.load(wT_ptr + bin_chain[:, None] * C_out + c[None, :])

        if has_token_feat:
            delta = w_pos + w_tok + same_ent.to(tl.float32)[:, None] * w_ent_row[None, :] + w_chain
        else:
            delta = w_pos + same_ent.to(tl.float32)[:, None] * w_ent_row[None, :] + w_chain

        z_base = (b * N_rows * N + i_local * N + j_offs) * C_out
        z_old = tl.load(z_ptr + z_base[:, None] + c[None, :], mask[:, None], 0.0)
        tl.store(z_ptr + z_base[:, None] + c[None, :], z_old + delta, mask[:, None])


def fused_relpos_embed(
    z: torch.Tensor,
    res_idx: torch.Tensor,
    tok_idx: torch.Tensor | None,
    sym_id: torch.Tensor,
    asym_id: torch.Tensor,
    entity_id: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_relative_idx: int = 32,
    max_relative_chain: int = 2,
    entity_chain_cond: bool = True,
    has_token_feat: bool = True,
    row_start: int = 0,
) -> torch.Tensor:
    """Fused relative-position embedding: compute bin indices and project in one pass.

    Supports two C_in layouts:

    * ``has_token_feat=True`` (default, OF3 / Protenix / Boltz-2)::

        rel_feat = relpos_complex(batch, max_relative_idx, max_relative_chain)
        z = z + F.linear(rel_feat, weight)

    * ``has_token_feat=False`` (OF2 multimer)::

        rel_feat = [rel_pos | same_entity | rel_chain]   # no rel_token column
        z = z + F.linear(rel_feat, weight)

    Neither path materialises the [B, N, N, C_in] intermediate.

    Args:
        z:
            [B, N_rows, N, C_out] float32.  Modified in-place and returned.
            ``N_rows`` may be less than ``N`` for chunked (row-sliced) encoding;
            set ``row_start`` to the corresponding global row offset.
        res_idx:    [B, N] int32  residue_index
        tok_idx:    [B, N] int32  token_index — pass None when has_token_feat=False
        sym_id:     [B, N] int32  sym_id
        asym_id:    [B, N] int32  asym_id
        entity_id:  [B, N] int32  entity_id
        weight:     [C_out, C_in] float32  linear weight (no bias)
        max_relative_idx:    clip radius for residue/token bins (default 32)
        max_relative_chain:  clip radius for chain bins (default 2)
        entity_chain_cond:
            When True (default) the chain bin is valid for same-entity pairs,
            matching OF3 / Protenix / OF2 multimer.  When False the chain bin
            is valid for cross-chain pairs, matching Boltz-2.
        has_token_feat:
            When True (default) the weight layout includes a ``rel_token``
            block (OF3 / Protenix / Boltz-2, C_in = 2*n_rel+1+n_chain).
            When False the ``rel_token`` block is absent (OF2 multimer,
            C_in = n_rel+1+n_chain).
        row_start:
            Global token index of the first row in ``z``.  Used with chunked
            encoding; pass 0 (default) for the dense (full N×N) case.

    Returns:
        ``z``  (same tensor, modified in-place)
    """
    if z.dtype != torch.float32 or not z.is_contiguous():
        raise ValueError(f"z must be float32 and contiguous, got dtype={z.dtype} contiguous={z.is_contiguous()}")
    if weight.dtype != torch.float32:
        raise ValueError(f"weight must be float32, got {weight.dtype}")

    B, N_rows, N, C_out = z.shape
    n_rel = 2 * max_relative_idx + 2
    n_chain = 2 * max_relative_chain + 2
    ent_col = 2 * n_rel if has_token_feat else n_rel
    C_in = ent_col + 1 + n_chain

    wT = weight.T.contiguous()
    if wT.shape != (C_in, C_out):
        raise ValueError(f"weight shape mismatch: expected [{C_out},{C_in}], got {weight.shape}")

    def _i32(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.int32).contiguous()

    tok_ptr = _i32(tok_idx) if tok_idx is not None else _i32(res_idx)

    _fused_relpos_embed_kernel[(B * N_rows,)](
        _i32(res_idx),
        tok_ptr,
        _i32(sym_id),
        _i32(asym_id),
        _i32(entity_id),
        wT,
        z,
        N,
        N_rows,
        row_start,
        C_out=C_out,
        n_rel=n_rel,
        rel_clip=max_relative_idx,
        n_chain=n_chain,
        chain_clip=max_relative_chain,
        entity_chain_cond=entity_chain_cond,
        has_token_feat=has_token_feat,
        BLOCK_J=64,
        num_warps=4,
    )
    return z
