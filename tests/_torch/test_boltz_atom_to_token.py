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
"""The prepared Boltz atom-to-token slots reproduce the one-hot contractions."""

import pytest
import torch

from bionemo_ir._torch.layers.transformers.atom import prepare_atom_to_token
from bionemo_ir.dsl_kernels.triton.atom_gather_kernel import reduce_atom_slots

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def one_hot_map(counts: list[int], n_atoms: int, n_tokens: int) -> torch.Tensor:
    owners = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    atom_to_token = torch.zeros(1, n_atoms, n_tokens)
    atom_to_token[0, torch.arange(owners.numel()), owners] = 1.0
    return atom_to_token.cuda()


def encoder_reference(features: torch.Tensor, atom_to_token: torch.Tensor) -> torch.Tensor:
    mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
    mean = mean.unsqueeze(1).repeat_interleave(features.shape[1], 1)
    return torch.einsum("bijd,bijk->bikd", features, mean)


def decoder_reference(tokens: torch.Tensor, atom_to_token: torch.Tensor) -> torch.Tensor:
    one_hot = atom_to_token.unsqueeze(1).repeat_interleave(tokens.shape[1], 1)
    return torch.einsum("bikj,bijd->bikd", one_hot, tokens)


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_prepared_slots_match_one_hot(multiplicity: int) -> None:
    torch.manual_seed(0)
    counts = [3, 1, 14, 2, 0, 5]
    atom_to_token = one_hot_map(counts, n_atoms=32, n_tokens=8)
    prepared = prepare_atom_to_token(atom_to_token)
    assert prepared is not None

    features = torch.randn(1, multiplicity, 32, 16, device="cuda")
    reduced = reduce_atom_slots(
        features, prepared["gather_index"], prepared["gather_mask"], prepared["gather_counts"], 8, 1e-6
    )
    torch.testing.assert_close(reduced, encoder_reference(features, atom_to_token), rtol=1e-6, atol=1e-6)

    tokens = torch.randn(1, multiplicity, 8, 16, device="cuda")
    owners = prepared["atom_to_token_index"][:, None, :, None].expand(-1, multiplicity, -1, 16)
    gathered = tokens.gather(2, owners) * prepared["atom_mask"][:, None, :, None]
    torch.testing.assert_close(gathered, decoder_reference(tokens, atom_to_token), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["two_hot", "scattered", "fractional", "cpu"])
def test_unsupported_maps_keep_dense_path(layout: str) -> None:
    atom_to_token = one_hot_map([2, 2, 1], n_atoms=6, n_tokens=3)
    if layout == "two_hot":
        atom_to_token[0, 0, 2] = 1.0
    elif layout == "scattered":
        atom_to_token[0, [0, 3]] = atom_to_token[0, [3, 0]]
    elif layout == "fractional":
        atom_to_token[0, 0, 0] = 0.5
    else:
        atom_to_token = atom_to_token.cpu()
    assert prepare_atom_to_token(atom_to_token) is None
