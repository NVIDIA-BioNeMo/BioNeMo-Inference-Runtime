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
"""Bitwise confidence metadata checks without model weights."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.modules.openfold3.utils.atomize_utils import (
    get_token_atom_index_offset,
    get_token_atom_index_offsets,
)
from bionemo_ir._torch.modules.openfold3.utils.token_atom_constants import atom_name_to_index_by_restype
from bionemo_ir._torch.modules.protenix.confidence import ProtenixConfidenceHead
from bionemo_ir._torch.modules.protenix.summary import ChainIndex

_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


class _GatherHead(nn.Module):
    """Exercise the real preparation and per-sample gather without heavy logits."""

    prepare = ProtenixConfidenceHead.prepare
    per_sample_logits = ProtenixConfidenceHead.per_sample_logits

    def __init__(self) -> None:
        """Initialize the small projections used during request preparation."""
        super().__init__()
        self.dtype = torch.float32
        self.input_strunk_ln = nn.Identity()
        self.linear_no_bias_s1 = nn.Linear(3, 4, bias=False)
        self.linear_no_bias_s2 = nn.Linear(3, 4, bias=False)

    def _per_sample(self, *args: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Return gathered coordinates in place of the four logit tensors."""
        x_rep = args[6]
        assert x_rep is not None
        return (x_rep,) * 4


def _gather_inputs(device: str, mask: torch.Tensor) -> tuple[_GatherHead, dict]:
    """Build the small request context consumed by the real gather method."""
    head = _GatherHead().to(device)
    inputs = torch.zeros(2, 4, 3, device=device)
    features = {
        "distogram_rep_atom_mask": mask.unsqueeze(0).to(device),
        "atom_to_token_idx": torch.zeros(1, mask.numel(), dtype=torch.long, device=device),
        "atom_to_tokatom_idx": torch.zeros(1, mask.numel(), dtype=torch.long, device=device),
    }
    context = head.prepare(features, inputs, inputs, torch.zeros(2, 4, 4, 4, device=device))
    return head, context


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("selection", ["mixed", "all", "empty"])
def test_representative_gather_matches_boolean_bits(device: str, selection: str) -> None:
    """Gather the same ordered coordinate bits as boolean indexing."""
    mask = torch.tensor([0.0, 0.5, 0.0, -1.0, 0.0, 1.0])
    if selection == "all":
        mask.fill_(1)
    elif selection == "empty":
        mask.zero_()
    head, context = _gather_inputs(device, mask)
    # Keep the atom axis noncontiguous and retain negative zero's sign bit.
    coords = torch.arange(2 * 3 * 12 * 3, device=device, dtype=torch.float32).reshape(2, 3, 12, 3)[..., ::2, :]
    coords[..., 1, 0] = -0.0
    for sample in range(coords.shape[-3]):
        actual = head.per_sample_logits(context, coords[:, sample])[0]
        expected = coords[:, sample, mask.bool().to(device), :]
        assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("atom_device", _DEVICES)
def test_packed_chain_index_matches_separate_row_reference(device: str, atom_device: str) -> None:
    """Match separate token/atom copies on the same or different devices."""
    # Fractional flags must be tested for nonzero before conversion to int64.
    asym_id = torch.tensor([8.5, -2.5, 8.1, 3.2], device=device)
    has_frame = torch.tensor([0.0, 0.5, -1.0, 0.0], device=device)
    ligand = torch.tensor([0.25, 0.0, 1.0, -0.5], device=device)
    owners = torch.tensor([2, 0, 3, 1, 1], device=atom_device)
    polymer = torch.tensor([0.0, 0.5, -1.0, 1.0, 0.0], device=atom_device)
    tokens_host = torch.stack([asym_id.long(), (has_frame != 0).long(), (ligand != 0).long()]).cpu()
    atoms_host = torch.stack([owners.long(), (polymer != 0).long()]).cpu()
    index = ChainIndex(asym_id, has_frame, ligand, owners, polymer)
    asym_host = torch.searchsorted(torch.unique(tokens_host[0]), tokens_host[0])
    frame_host, ligand_host = tokens_host[1].bool(), tokens_host[2].bool()
    atom_asym_host = asym_host[atoms_host[0]]
    assert index.n_chain == torch.unique(tokens_host[0]).numel()
    assert torch.equal(index.asym, asym_host.to(device))
    assert torch.equal(index.all_tokens().frames, frame_host.nonzero().flatten().to(device))
    for chain in range(index.n_chain):
        token_mask = asym_host == chain
        atom_mask = atom_asym_host == chain
        selection = index.chain(chain)
        assert torch.equal(selection.tokens, token_mask.nonzero().flatten().to(device))
        assert torch.equal(selection.frames, frame_host[token_mask].nonzero().flatten().to(device))
        assert selection.n_frame == int(frame_host[token_mask].sum())
        assert index.chain_has_frame[chain] == bool(frame_host[token_mask].any())
        assert index.chain_is_ligand[chain] == bool(ligand_host[token_mask].sum() >= token_mask.sum() // 2)
        assert torch.equal(index.chain_atoms(chain), atom_mask.nonzero().flatten().to(device))
        assert index.chain_atom_count[chain] == int(atom_mask.sum())
        assert index.chain_is_polymer[chain] == bool(atoms_host[1, atom_mask].any())
        for other in range(chain + 1, index.n_chain):
            pair = index.pair(chain, other)
            pair_mask = token_mask | (asym_host == other)
            assert torch.equal(pair.tokens, pair_mask.nonzero().flatten().to(device))
            assert torch.equal(pair.frames, frame_host[pair_mask].nonzero().flatten().to(device))
            pair_atoms = atom_mask | (atom_asym_host == other)
            assert torch.equal(index.pair_atoms(chain, other), pair_atoms.nonzero().flatten().to(device))


def _old_atom_lookup(atom_name: str, restype: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The original two-copy expressions, including their float casts."""
    table = atom_name_to_index_by_restype[atom_name]
    return (
        torch.einsum("...k,k->...", restype, torch.tensor(table["index"], device=restype.device).float()).long(),
        torch.einsum("...k,k->...", restype, torch.tensor(table["mask"], device=restype.device).float()).long(),
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("atom_name", tuple(atom_name_to_index_by_restype))
def test_packed_atom_lookup_matches_original(device: str, atom_name: str) -> None:
    """Keep the original lookup results for every residue and atom name."""
    width = len(atom_name_to_index_by_restype[atom_name]["index"])
    # Every residue row, with a batch and sample axis, exercises unknown/gap entries too.
    restype = F.one_hot(torch.arange(width, device=device).reshape(1, 1, width), width).float().expand(2, 3, -1, -1)
    actual = get_token_atom_index_offset(atom_name, restype)
    expected = _old_atom_lookup(atom_name, restype)
    assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("atom_names", [("CB", "CA", "C4", "C2"), ("N", "CA", "C", "C3'", "C1'", "C4'")])
def test_batched_atom_lookup_matches_original(device: str, atom_names: tuple[str, ...]) -> None:
    """Match one original lookup per name, for the representative-atom and frame name sets."""
    width = len(atom_name_to_index_by_restype[atom_names[0]]["index"])
    # Every residue row plus all-zero padding rows, with a batch and sample axis.
    restype = F.one_hot(torch.arange(width, device=device), width).float()
    restype = torch.cat([restype, torch.zeros(3, width, device=device)]).expand(2, 3, -1, -1)
    offsets, masks = get_token_atom_index_offsets(atom_names, restype)
    for column, atom_name in enumerate(atom_names):
        expected_offset, expected_mask = _old_atom_lookup(atom_name, restype)
        assert torch.equal(offsets[..., column], expected_offset)
        assert torch.equal(masks[..., column], expected_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_representative_gather_graph_replay() -> None:
    """Reuse representative indices while graph replay reads fresh coordinates."""
    mask = torch.tensor([0, 1, 0, 1, 0, 1])
    head, context = _gather_inputs("cuda", mask)
    coords = torch.arange(2 * 6 * 3, device="cuda", dtype=torch.float32).reshape(2, 6, 3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        head.per_sample_logits(context, coords)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = head.per_sample_logits(context, coords)[0]
    for _ in range(3):
        coords.add_(1)
        graph.replay()
        assert torch.equal(actual, coords[:, mask.bool().cuda(), :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_atom_lookup_graph_replay_and_streams() -> None:
    """Share the cached device lookup tables across streams and graph replays."""
    atom_name = "CA"
    width = len(atom_name_to_index_by_restype[atom_name]["index"])
    restype = torch.eye(width, device="cuda").unsqueeze(0)
    # The first call uploads the device tables synchronously, so it must precede capture.
    expected = _old_atom_lookup(atom_name, restype)
    get_token_atom_index_offset(atom_name, restype)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs.append(get_token_atom_index_offset(atom_name, restype))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for output in outputs:
        assert all(torch.equal(a, e) for a, e in zip(output, expected, strict=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = get_token_atom_index_offset(atom_name, restype)
    for _ in range(3):
        graph.replay()
        assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
