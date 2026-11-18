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

# Copyright 2025 AlQuraishi Laboratory
# Modified by NVIDIA Corporation and affiliates.

import torch


def relpos_complex(
    batch: dict[str, torch.Tensor],
    max_relative_idx: int,
    max_relative_chain: int,
    *,
    row_slice: slice | None = None,
) -> torch.Tensor:
    """
    Args:
        batch:
            Input feature dictionary with integer residue, token, chain,
            entity, and symmetry indices of shape [*, N_token].
        max_relative_idx:
            Maximum relative position and token indices clipped
        max_relative_chain:
            Maximum relative chain indices clipped
        row_slice:
            Optional query-token rows to encode against all key tokens.

    Returns:
        [*, N_query, N_token, C_z] fp32 relative-position features. N_query
        equals N_token unless row_slice selects a subset of query rows.
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    rows = slice(None) if row_slice is None else row_slice
    same_chain = asym_id[..., rows, None] == asym_id[..., None, :]
    same_res = res_idx[..., rows, None] == res_idx[..., None, :]
    same_entity = entity_id[..., rows, None] == entity_id[..., None, :]

    def relpos(pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int) -> torch.Tensor:
        """
        Args:
            pos:
                [*, N_token] Token index
            condition:
                [*, N_query, N_token] Condition for clipping
            rel_clip_idx:
                Max idx for clipping (max_relative_idx or max_relative_chain)
        Returns:
            rel_pos:
                [*, N_query, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., rows, None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            torch.full_like(clipped_offset, 2 * rel_clip_idx + 1),
        )
        # These clipped integer indices already identify their nearest bin.
        # Scatter directly into fp32 instead of constructing bin-distance and
        # one-hot int64 tensors with a full trailing bin dimension.
        rel_pos = torch.zeros(
            (*final_offset.shape, 2 * rel_clip_idx + 2), device=final_offset.device, dtype=torch.float32
        )
        return rel_pos.scatter_(-1, final_offset.long().unsqueeze(-1), 1.0)

    rel_pos = relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

    rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1)

    return rel_feat
