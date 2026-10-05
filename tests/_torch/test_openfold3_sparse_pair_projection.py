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

"""Sparse pair conditioning preserves local atom windows."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.layers.sequence_local_atom import create_gather_indices, query_to_keys_optimized
from bionemo_ir._torch.modules.openfold3.sequence_local_atom_attention import (
    NoisyPositionEmbedder,
    convert_pair_atom_to_blocks,
)
from bionemo_ir.dsl_kernels.triton.sparse_pair_projection import _cached_projections, project_pair_windows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def pair_inputs(tokens: int, atoms: int, batch: int = 1) -> tuple:
    torch.manual_seed(13)
    owner = (torch.arange(atoms, device="cuda") // 8).clamp_max(tokens - 1).expand(batch, -1).clone()
    owner[:, ::11] = torch.randint(0, tokens, owner[:, ::11].shape, device="cuda")
    mask = torch.randint(0, 2, (batch, atoms), device="cuda").float()
    indices, _ = create_gather_indices(atoms // 32 + 1, 32, 128, torch.device("cuda"))
    metadata = AttentionMetadata(query_to_keys=partial(query_to_keys_optimized, gather_indices=indices, W=32, H=128))
    return {"atom_to_token_index": owner, "atom_mask": mask}, metadata


@pytest.mark.parametrize("atoms", [32, 65, 257])
@pytest.mark.parametrize("sample_dim", [False, True])
@torch.no_grad()
def test_pair_projection(atoms: int, sample_dim: bool) -> None:
    tokens = 64
    batch, metadata = pair_inputs(tokens, atoms, 2)
    pair = torch.randn(2, tokens, tokens, 128, device="cuda")
    gamma = torch.randn(128, device="cuda")
    weight = torch.randn(16, 128, device="cuda") * 0.1
    if sample_dim:
        batch["atom_to_token_index"] -= tokens
        pair = pair.unsqueeze(0)
        batch = {name: value.unsqueeze(0) for name, value in batch.items()}
    dense = F.linear(F.layer_norm(pair, (128,), gamma, eps=1e-5), weight)
    expected = convert_pair_atom_to_blocks(batch, dense, 32, 128, metadata)
    project = partial(project_pair_windows, norm_weight=gamma, weight=weight, eps=1e-5)
    actual = convert_pair_atom_to_blocks(batch, pair, 32, 128, metadata, project=project)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual[expected == 0], expected[expected == 0])
    # An out-of-range token zeroes its pairs, as masking its atom does.
    masked = dict(batch, atom_mask=batch["atom_mask"].clone())
    masked["atom_mask"][..., -1] = 0
    expected = convert_pair_atom_to_blocks(masked, dense, 32, 128, metadata)
    batch["atom_to_token_index"][..., -1] = tokens
    actual = convert_pair_atom_to_blocks(batch, pair, 32, 128, metadata, project=project)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tokens", [32, 128, 512, 1024])
@torch.no_grad()
def test_noisy_embedder(tokens: int, monkeypatch: pytest.MonkeyPatch) -> None:
    atoms = 4 * tokens + 1
    batch, metadata = pair_inputs(tokens, atoms)
    batch["token_mask"] = torch.ones(1, tokens, device="cuda")
    batch["num_atoms_per_token"] = torch.bincount(batch["atom_to_token_index"][0], minlength=tokens).unsqueeze(0)
    batch["atom_broadcast_index"] = batch["atom_to_token_index"].flatten()
    model = NoisyPositionEmbedder(32, 128, 32, 16).cuda().eval()
    for parameter in model.parameters():
        parameter.normal_(std=0.1)
    pair = torch.randn(1, tokens, tokens, 128, device="cuda")
    single = torch.randn(1, tokens, 32, device="cuda")
    coords = torch.randn(1, atoms, 3, device="cuda")
    cl = torch.randn(1, atoms, 32, device="cuda")
    plm = torch.randn(1, atoms // 32 + 1, 32, 128, 16, device="cuda")
    assert model._can_project_sparse(pair)
    actual = model(batch, cl, plm, single, pair, coords, 32, 128, metadata)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
        expected = model(batch, cl, plm, single, pair, coords, 32, 128, metadata)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    # The captured graph replays the sparse kernel, so it matches eager exactly.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model(batch, cl, plm, single, pair, coords, 32, 128, metadata)
    graph.replay()
    for got, ref in zip(captured, actual, strict=True):
        assert torch.equal(got, ref)


@pytest.mark.parametrize("launch", ["driver", "compiled", "unaligned"])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bool])
@torch.no_grad()
def test_cached_launch(launch: str, mask_dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    kernels = _cached_projections(torch.cuda.current_device(), mask_dtype)
    if launch == "driver" and kernels.project.driver is None:
        pytest.skip("CUDA driver launcher unavailable")
    if launch == "compiled":
        monkeypatch.setattr(kernels.project, "_driver", None)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for rows, eps, repeats in (
            (0, 1e-5, 1),
            (1, 1e-5, 1),
            (35, 0.01, 1),
            (259, 1e-5, 8),
            (259, 1e-5, 5),
            (259, 1e-5, 1),
        ):
            offset = int(launch == "unaligned")
            pair = torch.randn(8 * 8 * 128 + offset, device="cuda")[offset:].view(1, 8, 8, 128)
            gamma = torch.randn(128, device="cuda")
            weight = torch.randn(16, 128, device="cuda") * 0.1
            indices = (torch.arange(rows, device="cuda") // repeats % 64).view(1, 1, 1, rows)
            mask = torch.randint(0, 2, (rows + offset,), device="cuda").to(mask_dtype)[offset:]
            expected = F.linear(F.layer_norm(pair, (128,), gamma, eps=eps), weight).view(-1, 16)[indices.flatten()]
            expected = (expected * mask[:, None]).view(1, 1, 1, rows, 16)
            actual = project_pair_windows(pair, indices, mask, norm_weight=gamma, weight=weight, eps=eps)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    stream.synchronize()


@torch.no_grad()
def test_bounds_unsafe_inputs() -> None:
    pair = torch.randn(1, 4, 4, 128, device="cuda")
    indices = torch.arange(3, device="cuda").view(1, 1, 1, 3)
    mask = torch.ones(1, 1, 1, 3, device="cuda")
    gamma = torch.randn(128, device="cuda")
    weight = torch.randn(16, 128, device="cuda")
    project = partial(project_pair_windows, norm_weight=gamma, weight=weight, eps=1e-5)
    dense = F.linear(F.layer_norm(pair, (128,), gamma, eps=1e-5), weight).view(-1, 16)
    # Addresses outside the 16 pairs project to zero instead of reading out of bounds.
    for addresses in (indices + 14, indices - 1):
        flat = addresses.flatten()
        inside = (flat >= 0) & (flat < 16)
        expected = torch.where(inside[:, None], dense[flat.clamp(0, 15)], 0).view(1, 1, 1, 3, 16)
        torch.testing.assert_close(project(pair, addresses, mask), expected, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        project(pair, indices, mask[..., :2])
    with pytest.raises(ValueError):
        project(pair.transpose(1, 2), indices, mask)
