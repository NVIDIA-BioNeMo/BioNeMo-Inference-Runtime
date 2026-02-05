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

from dataclasses import dataclass

import pytest
import torch
from test_utils.openfold3.atom_attention_block_utils import \
    convert_trunk_pair_rep_to_blocks

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)
from tensorrt_bionemo._torch.modules.openfold3.sequence_local_atom_attention import \
    convert_pair_atom_to_blocks


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_tokens: int = 76
    n_atoms: int = 601
    n_queries: int = 32
    n_keys: int = 128
    n_dims: int = 16
    torch_dtype: torch.dtype = torch.float32


@pytest.mark.parametrize("sc", [
    Scenario(torch_dtype=torch.float32),
    Scenario(torch_dtype=torch.bfloat16),
])
def test_query_to_keys(sc: Scenario):

    K = (sc.n_atoms + sc.n_queries - 1) // sc.n_queries
    W = sc.n_queries
    H = sc.n_keys

    device = torch.device('cuda')
    keys_indexing_matrix = create_indexing_matrix(K, W, H, device)
    to_keys = lambda x: query_to_keys(x, keys_indexing_matrix, W, H)

    attn_metadata = AttentionMetadata(
        query_to_keys=to_keys,
        bias_cache={},
    )

    atom_to_token_index = torch.repeat_interleave(
        torch.arange(sc.n_tokens),
        repeats=8)[:sc.n_atoms].unsqueeze(0).unsqueeze(0).to(device)
    atom_mask = torch.randint(0,
                              2, (1, 1, sc.n_atoms),
                              dtype=sc.torch_dtype,
                              device=device)
    zij_trunk = torch.randn((1, 1, sc.n_tokens, sc.n_tokens, sc.n_dims),
                            dtype=sc.torch_dtype,
                            device=device)

    output = convert_pair_atom_to_blocks(
        zij_trunk=zij_trunk,
        atom_to_token_index=atom_to_token_index,
        atom_mask=atom_mask,
        n_query=sc.n_queries,
        n_key=sc.n_keys,
        attn_metadata=attn_metadata,
    )

    batch = {
        "atom_to_token_index": atom_to_token_index,
        "atom_mask": atom_mask,
    }
    ref_output = convert_trunk_pair_rep_to_blocks(batch=batch,
                                                  zij_trunk=zij_trunk,
                                                  n_query=sc.n_queries,
                                                  n_key=sc.n_keys)
    assert torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
