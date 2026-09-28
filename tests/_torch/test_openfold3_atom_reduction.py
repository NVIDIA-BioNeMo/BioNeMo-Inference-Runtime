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

from bionemo_ir._torch.layers.sequence_local_atom import aggregate_atom_features_to_tokens
from bionemo_ir._torch.modules.openfold3.utils.atomize_utils import (
    aggregate_atom_feat_to_tokens,
    prepare_atom_reduction,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def make_batch(device: str, mask_dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    return {
        "token_mask": torch.ones(2, 3, device=device),
        "num_atoms_per_token": torch.tensor([[2, 1, 0], [1, 2, 0]], device=device),
        "start_atom_index": torch.tensor([[0, 2, 0], [0, 1, 0]], device=device),
        "atom_to_token_index": torch.tensor([[0, 0, 1, 99, 99], [0, 1, 1, 99, 99]], device=device),
        "atom_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], device=device, dtype=mask_dtype),
    }


def reduce_prepared(batch: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    return aggregate_atom_feat_to_tokens(
        token_mask=batch["token_mask"],
        atom_to_token_index=batch["atom_to_token_index"],
        atom_mask=batch["atom_mask"],
        atom_feat=features,
        atom_dim=-2,
        gather_index=batch.get("atom_gather_index"),
        gather_mask=batch.get("atom_gather_mask"),
        gather_counts=batch.get("atom_gather_counts"),
        num_atoms_per_token=batch["num_atoms_per_token"],
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("present", range(1, 7))
@pytest.mark.parametrize("layout", ["valid", "invalid", "missing"])
def test_partial_metadata_preparation(device: str, present: int, layout: str) -> None:
    batch = make_batch(device)
    metadata = prepare_atom_reduction(batch)
    if layout == "invalid":
        batch["start_atom_index"][0, 1] = 0
    elif layout == "missing":
        del batch["start_atom_index"]
    features = torch.randn(2, 5, 8, device=device)
    expected = reduce_prepared(batch, features)
    keys = ("atom_gather_index", "atom_gather_mask", "atom_gather_counts")
    partial = {key: metadata[key] for bit, key in enumerate(keys) if present & (1 << bit)}
    batch.update(partial)

    prepared = prepare_atom_reduction(batch)

    assert all((key in prepared) == (layout == "valid") for key in keys)
    assert torch.equal(reduce_prepared(prepared, features), expected)
    assert all(batch[key] is value for key, value in partial.items())
    assert all((key in batch) == (key in partial) for key in keys)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bool])
def test_prepared_masked_batches(device: str, sampled: bool, mask_dtype: torch.dtype) -> None:
    batch = make_batch(device, mask_dtype)
    batch["atom_mask"][0, 1] = 0
    prepared = prepare_atom_reduction(batch)
    assert "atom_gather_index" not in batch
    assert prepare_atom_reduction(prepared) is prepared
    if sampled:
        batch = {key: value.unsqueeze(1) for key, value in batch.items()}
        prepared = {key: value.unsqueeze(1) for key, value in prepared.items()}
    shape = (2, 3, 5, 260) if sampled else (2, 5, 260)
    features = torch.randn(shape, device=device)[..., ::2]
    features.masked_fill_(~batch["atom_mask"].bool().unsqueeze(-1), float("nan"))
    expected = reduce_prepared(batch, features)
    actual = reduce_prepared(prepared, features)
    assert torch.equal(actual, expected)
    assert actual.isfinite().all()
    assert actual.shape[-1] == 130
    assert torch.count_nonzero(actual[..., 2, :]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_distinct_batch_denominators(device: str) -> None:
    batch = prepare_atom_reduction(make_batch(device))
    batch = {key: value.unsqueeze(1) for key, value in batch.items()}
    features = torch.tensor([2.0, 4.0, 8.0, 0.0, 0.0, 10.0, 20.0, 30.0, 0.0, 0.0], device=device).reshape(2, 1, 5, 1)
    features = features.expand(2, 3, 5, 1)
    actual = reduce_prepared(batch, features)
    expected = torch.tensor([3.0, 8.0, 0.0, 10.0, 25.0, 0.0], device=device).reshape(2, 1, 3, 1).expand(2, 3, 3, 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_precision_falls_back(device: str, dtype: torch.dtype, mask_dtype: torch.dtype) -> None:
    batch = make_batch(device)
    prepared = prepare_atom_reduction(batch)
    batch["atom_mask"] = batch["atom_mask"].to(mask_dtype)
    prepared["atom_mask"] = batch["atom_mask"]
    batch = {key: value.unsqueeze(1) for key, value in batch.items()}
    prepared = {key: value.unsqueeze(1) for key, value in prepared.items()}
    features = torch.randn(2, 3, 5, 130, device=device, dtype=dtype)
    expected = reduce_prepared(batch, features)
    actual = reduce_prepared(prepared, features)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert actual.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("atoms", [0, 1, 30, 31, 32, 33, 128])
def test_segment_order_boundary(device: str, atoms: int) -> None:
    batch = {
        "token_mask": torch.ones(1, 2, device=device),
        "num_atoms_per_token": torch.tensor([[atoms, 0]], device=device),
        "start_atom_index": torch.zeros(1, 2, dtype=torch.long, device=device),
        "atom_to_token_index": torch.zeros(1, atoms + 2, dtype=torch.long, device=device),
        "atom_mask": torch.cat((torch.ones(1, atoms, device=device), torch.zeros(1, 2, device=device)), dim=-1),
    }
    prepared = prepare_atom_reduction(batch)
    assert ("atom_gather_index" in prepared) == (atoms < 32)
    features = torch.randn(1, atoms + 2, 130, device=device)
    assert torch.equal(reduce_prepared(prepared, features), reduce_prepared(batch, features))
    sampled = {key: value.unsqueeze(1) for key, value in prepared.items()}
    reference = {key: value.unsqueeze(1) for key, value in batch.items()}
    features = features.unsqueeze(1).expand(1, 3, atoms + 2, 130)
    assert torch.equal(reduce_prepared(sampled, features), reduce_prepared(reference, features))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("invalid", ["owners", "starts", "counts", "padding", "fractional", "missing"])
def test_invalid_layout_fallback(device: str, invalid: str) -> None:
    batch = make_batch(device)
    if invalid == "owners":
        batch["atom_to_token_index"][0, :3] = torch.tensor([1, 0, 0], device=device)
    elif invalid == "starts":
        batch["start_atom_index"][1, 1] = 100
    elif invalid == "counts":
        batch["num_atoms_per_token"][0, 0] = -1
    elif invalid == "padding":
        batch["atom_mask"][0, -1] = 1
    elif invalid == "fractional":
        batch["atom_mask"][0, 1] = 0.5
    else:
        del batch["start_atom_index"]
    assert prepare_atom_reduction(batch) is batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sampled", [False, True])
def test_prepared_graph_replay(sampled: bool) -> None:
    batch = prepare_atom_reduction(make_batch("cuda"))
    if sampled:
        batch = {key: value.unsqueeze(1) for key, value in batch.items()}
    shape = (2, 3, 5, 130) if sampled else (2, 5, 130)
    features = torch.randn(shape, device="cuda")
    reduce_prepared(batch, features)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reduce_prepared(batch, features)
    reference = {key: value for key, value in batch.items() if not key.startswith("atom_gather_")}
    for _ in range(3):
        features.add_(1)
        expected = reduce_prepared(reference, features)
        graph.replay()
        assert torch.equal(output, expected)


def test_partial_metadata_rejected() -> None:
    with pytest.raises(ValueError, match="mask and counts"):
        aggregate_atom_features_to_tokens(
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1),
            torch.ones(1, 1, 1),
            atom_dim=-2,
            gather_index=torch.zeros(1, 1, dtype=torch.long),
        )
