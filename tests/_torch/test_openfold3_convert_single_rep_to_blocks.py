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
    convert_single_rep_to_blocks

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atoms: int = 608
    n_queries: int = 32
    n_keys: int = 128
    torch_dtype: torch.dtype = torch.float32


@pytest.mark.parametrize("sc", [
    Scenario(torch_dtype=torch.float64),
    Scenario(torch_dtype=torch.float32),
    Scenario(torch_dtype=torch.bfloat16),
    Scenario(n_atoms=1024),
])
def test_query_to_keys(sc: Scenario):

    K = sc.n_atoms // sc.n_queries
    W = sc.n_queries
    H = sc.n_keys

    device = torch.device('cuda')
    keys_indexing_matrix = create_indexing_matrix(K, W, H, device)
    to_keys = lambda x: query_to_keys(x, keys_indexing_matrix, W, H)

    attn_metadata = AttentionMetadata(
        query_to_keys=to_keys,
        bias_cache={},
    )

    ref_pos = torch.randn((1, K, W, 64), dtype=sc.torch_dtype, device=device)
    atom_mask = torch.randint(0,
                              2, (1, K, W),
                              dtype=sc.torch_dtype,
                              device=device)
    output = attn_metadata.query_to_keys(ref_pos.flatten(1, 2)).squeeze(1)
    output_atom_mask = atom_mask.unsqueeze(-1) * attn_metadata.query_to_keys(
        atom_mask.unsqueeze(-1)).squeeze(1).squeeze(-1).unsqueeze(-2)

    d_l, d_m, atom_mask = convert_single_rep_to_blocks(
        ql=ref_pos.flatten(1, 2),
        n_query=sc.n_queries,
        n_key=sc.n_keys,
        atom_mask=atom_mask.flatten(1, 2),
    )

    assert torch.allclose(output, d_m, atol=1e-2, rtol=1e-2)
    assert torch.allclose(output_atom_mask, atom_mask, atol=1e-2, rtol=1e-2)
