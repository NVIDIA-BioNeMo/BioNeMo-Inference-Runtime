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

import torch

from bionemo_ir._torch.layers.sequence_local_atom import (
    aggregate_atom_features_to_tokens,
    aggregate_indexed_atom_features,
    broadcast_token_features_to_atoms,
    compute_atom_broadcast_index,
    gather_token_features_to_atoms,
    select_atoms_from_padded_tokens,
)


def test_ragged_token_atom_broadcast_static_index_matches_dynamic_path():
    token_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    atom_counts = torch.tensor([[2, 1, 4], [1, 2, 1]])
    token_features = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    expand_index = compute_atom_broadcast_index(token_mask, atom_counts)

    dynamic = broadcast_token_features_to_atoms(token_mask, atom_counts, token_features)
    indexed = broadcast_token_features_to_atoms(
        token_mask,
        atom_counts,
        token_features,
        expand_index=expand_index,
    )

    assert torch.equal(indexed, dynamic)
    assert torch.equal(dynamic, torch.tensor([[10.0, 10.0, 20.0, 0.0], [40.0, 50.0, 50.0, 60.0]]))


def test_ragged_token_atom_broadcast_sample_axis_keeps_batch_major_counts():
    token_mask = torch.ones(2, 3, dtype=torch.bool)
    atom_counts = torch.tensor([[2, 1, 0], [1, 0, 0]])
    token_features = torch.tensor(
        [
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [[40.0, 50.0, 60.0], [41.0, 51.0, 61.0]],
        ]
    )
    expand_index = compute_atom_broadcast_index(token_mask, atom_counts)

    dynamic = broadcast_token_features_to_atoms(token_mask, atom_counts, token_features)
    indexed = broadcast_token_features_to_atoms(
        token_mask,
        atom_counts,
        token_features,
        expand_index=expand_index,
    )

    expected = torch.tensor(
        [
            [[10.0, 10.0, 20.0], [11.0, 11.0, 21.0]],
            [[40.0, 0.0, 0.0], [41.0, 0.0, 0.0]],
        ]
    )
    assert torch.equal(dynamic, expected)
    assert torch.equal(indexed, expected)


def test_ragged_atom_reduction_excludes_masked_atoms_from_mean():
    token_mask = torch.tensor([[True, True]])
    atom_to_token = torch.tensor([[0, 0, 1, 1]])
    atom_mask = torch.tensor([[True, False, True, True]])
    atom_features = torch.tensor([[[2.0], [100.0], [6.0], [10.0]]])

    reduced = aggregate_atom_features_to_tokens(
        token_mask,
        atom_to_token,
        atom_mask,
        atom_features,
        atom_dim=-2,
    )

    assert torch.equal(reduced, torch.tensor([[[2.0], [8.0]]]))


def test_indexed_gather_and_reduction_make_denominator_semantics_explicit():
    token_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    atom_to_token = torch.tensor([[0, 0, 1]])
    gathered = gather_token_features_to_atoms(token_features, atom_to_token)
    assert torch.equal(
        gathered,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]]),
    )

    atom_features = torch.tensor([[[2.0], [100.0], [6.0]]])
    atom_mask = torch.tensor([[True, False, True]])
    excluded = aggregate_indexed_atom_features(
        atom_features,
        atom_to_token,
        2,
        atom_mask=atom_mask,
        count_mask=atom_mask,
        deterministic=False,
    )
    included = aggregate_indexed_atom_features(
        atom_features,
        atom_to_token,
        2,
        atom_mask=atom_mask,
        deterministic=False,
    )
    assert torch.equal(excluded, torch.tensor([[[2.0], [6.0]]]))
    assert torch.equal(included, torch.tensor([[[1.0], [6.0]]]))


def test_indexed_topology_broadcasts_over_sample_axis():
    token_features = torch.tensor([[[[1.0], [3.0]], [[2.0], [4.0]]]])
    atom_to_token = torch.tensor([[0, 0, 1]])

    gathered = gather_token_features_to_atoms(token_features, atom_to_token)
    assert torch.equal(
        gathered,
        torch.tensor([[[[1.0], [1.0], [3.0]], [[2.0], [2.0], [4.0]]]]),
    )
    reduced = aggregate_indexed_atom_features(
        gathered,
        atom_to_token,
        2,
        deterministic=False,
    )
    assert torch.equal(reduced, token_features)


def test_select_atoms_from_padded_tokens_flattens_mask_with_batch():
    features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    selected = select_atoms_from_padded_tokens(features, mask)
    assert torch.equal(
        selected,
        torch.tensor(
            [
                [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]],
                [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            ]
        ),
    )

    features_2d = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    mask_2d = torch.tensor(
        [
            [[1, 0, 1, 0], [1, 1, 0, 0]],
            [[1, 0, 0, 0], [1, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    selected_2d = select_atoms_from_padded_tokens(features_2d, mask_2d)
    zeros = torch.zeros(3)
    assert torch.equal(
        selected_2d,
        torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([features_2d[0, 0, 0], features_2d[0, 0, 2], zeros]),
                        torch.stack([features_2d[0, 1, 0], features_2d[0, 1, 1], zeros]),
                    ]
                ),
                torch.stack(
                    [
                        torch.stack([features_2d[1, 0, 0], zeros, zeros]),
                        torch.stack([features_2d[1, 1, 0], features_2d[1, 1, 1], features_2d[1, 1, 2]]),
                    ]
                ),
            ]
        ),
    )

    features_flat = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    mask_flat = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    selected_flat = select_atoms_from_padded_tokens(features_flat, mask_flat)
    assert torch.equal(selected_flat, torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]))


def test_select_atoms_from_padded_tokens_tiles_mask_across_sample_axis():
    features = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    selected = select_atoms_from_padded_tokens(features, mask)
    assert torch.equal(
        selected,
        torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([features[0, 0, 0], features[0, 0, 2]]),
                        torch.stack([features[0, 1, 0], features[0, 1, 2]]),
                    ]
                ),
                torch.stack(
                    [
                        torch.stack([features[1, 0, 0], features[1, 0, 1]]),
                        torch.stack([features[1, 1, 0], features[1, 1, 1]]),
                    ]
                ),
            ]
        ),
    )

    features_b1 = features[:1]
    mask_b1 = mask[:1]
    selected_b1 = select_atoms_from_padded_tokens(features_b1, mask_b1)
    assert torch.equal(selected_b1, selected[:1])
