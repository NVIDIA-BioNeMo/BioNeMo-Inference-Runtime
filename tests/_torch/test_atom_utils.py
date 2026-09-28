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

import pytest
import torch

from bionemo_ir._torch.layers.sequence_local_atom import (
    aggregate_atom_features_to_tokens,
    aggregate_indexed_atom_features,
    broadcast_token_features_to_atoms,
    compute_atom_broadcast_index,
    gather_token_features_to_atoms,
    prepare_indexed_reduction,
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


def test_gather_atom_reduction_matches_masked_mean_with_sample_axis():
    token_mask = torch.ones((1, 1, 2), dtype=torch.bool)
    atom_to_token = torch.tensor([[[0, 0, 1, 1]]])
    atom_mask = torch.tensor([[[True, False, True, True]]])
    atom_features = torch.arange(60.0).reshape(1, 5, 4, 3)
    gather_index = torch.tensor([[[0, 1, 2, 3]]])
    gather_mask = torch.tensor([[[[True, False], [True, True]]]])
    gather_counts = torch.tensor([[[1, 2]]])
    kwargs = {
        "token_mask": token_mask,
        "atom_to_token_index": atom_to_token,
        "atom_mask": atom_mask,
        "atom_features": atom_features,
        "atom_dim": -2,
    }
    reference = aggregate_atom_features_to_tokens(**kwargs)
    gathered = aggregate_atom_features_to_tokens(
        **kwargs,
        gather_index=gather_index,
        gather_mask=gather_mask,
        gather_counts=gather_counts,
    )
    assert torch.equal(gathered, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size,channels", [(1, 768), (2, 130)])
def test_gather_atom_reduction_matches_deterministic_scatter_on_cuda(batch_size, channels):
    torch.manual_seed(7)
    token_mask = torch.ones((batch_size, 1, 2), dtype=torch.bool, device="cuda")
    atom_to_token = torch.tensor([[[0] * 30 + [1] * 30]], device="cuda").expand(batch_size, -1, -1)
    atom_mask = torch.ones((batch_size, 1, 60), dtype=torch.bool, device="cuda")
    atom_mask[0, 0, 5] = False
    if batch_size == 2:
        atom_mask[1, 0, 33] = False
    atom_features = torch.randn((batch_size, 5, 60, channels), device="cuda")
    gather_index = torch.arange(60, device="cuda").reshape(1, 1, 60).expand(batch_size, -1, -1)
    gather_mask = atom_mask.reshape(batch_size, 1, 2, 30)
    gather_counts = gather_mask.sum(-1)
    kwargs = {
        "token_mask": token_mask,
        "atom_to_token_index": atom_to_token,
        "atom_mask": atom_mask,
        "atom_features": atom_features,
        "atom_dim": -2,
    }
    reference = aggregate_atom_features_to_tokens(**kwargs)
    gathered = aggregate_atom_features_to_tokens(
        **kwargs,
        gather_index=gather_index,
        gather_mask=gather_mask,
        gather_counts=gather_counts,
    )
    assert torch.equal(gathered, reference)


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


@pytest.mark.parametrize("owners", [[[1, 0]], [[0, -1]], [[0, 3]], [[0] * 32]])
def test_indexed_reduction_fallback(owners: list[list[int]]) -> None:
    assert prepare_indexed_reduction(torch.tensor(owners), 3) is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_protenix_prepared_reduction(device: str, dtype: torch.dtype) -> None:
    from bionemo_ir._torch.modules.protenix.atom_attention import _aggregate_atom_to_token

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    owners = torch.tensor([[0, 0, 2, 2, 2], [0, 1, 1, 1, 1]], device=device)
    metadata = prepare_indexed_reduction(owners, 4)
    assert metadata is not None
    samples = 3
    expanded = owners[:, None].expand(-1, samples, -1).reshape(6, 5)
    if dtype == torch.bfloat16:
        # Exact sums isolate fallback dispatch from atomics.
        features = torch.randint(-8, 9, (6, 5, 130), device=device).to(dtype)
    else:
        features = torch.randn(6, 5, 130, device=device, dtype=dtype)
    reduce = _aggregate_atom_to_token
    expected = reduce(features, expanded, 4)
    actual = reduce(features, expanded, 4, metadata)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, 3]) == 0
    if device == "cuda" and dtype == torch.float32:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = reduce(features, expanded, 4, metadata)
        features.mul_(2)
        graph.replay()
        torch.testing.assert_close(captured, expected * 2)
