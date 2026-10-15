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

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.openfold3.common import encode_atom_name_chars as encode_atom_name_chars_openfold3
from bionemo_ir.pipeline.utils.atom import (
    ATOM_NAME_TO_FOLDING_INDEX,
    NUM_FOLDING_ATOM_TYPES,
    decode_atom_name_chars,
    decode_element_symbols,
    encode_atom_name_chars,
    encode_atom_name_chars_one_hot,
    scatter_flat_atoms_to_folding_layout,
)


def test_atom_name_codec_is_shared_with_openfold3():
    assert encode_atom_name_chars("CA") == [35, 33, 0, 0]
    assert encode_atom_name_chars_openfold3("CA") == encode_atom_name_chars("CA")

    names = ["CA", "C4'", "ZN", "PAD"]
    encoded = encode_atom_name_chars_one_hot(names)
    decoded = decode_atom_name_chars(
        encoded,
        torch.tensor([True, True, True, False]),
    )
    assert decoded.tolist() == ["CA", "C4'", "ZN", ""]


def test_empty_atom_name_batch_preserves_codec_rank():
    encoded = encode_atom_name_chars_one_hot([])
    assert encoded.shape == (0, 4, 64)
    assert encoded.dtype == torch.int32


def test_scatter_flat_atoms_to_folding_layout():
    positions = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    ownership = np.array([0, 1, 2])
    names = ["CA", "N", "UNKNOWN"]
    active = np.array([True, True, True])

    output, mask, _, _ = scatter_flat_atoms_to_folding_layout(
        positions,
        ownership,
        names,
        active,
        num_tokens=2,
    )

    assert output.shape == (2, NUM_FOLDING_ATOM_TYPES, 3)
    assert mask.shape == (2, NUM_FOLDING_ATOM_TYPES)
    assert np.array_equal(
        output[0, ATOM_NAME_TO_FOLDING_INDEX["CA"]],
        positions[0],
    )
    assert np.array_equal(
        output[1, ATOM_NAME_TO_FOLDING_INDEX["N"]],
        positions[1],
    )
    assert mask.sum() == 2


def test_scatter_preserves_atoms_outside_the_atom_types_universe():
    """Regression: SMILES ligand atoms were silently dropped (nvbug 6932678).

    ``_build_smiles_mol`` names atoms ``SYMBOL + canonical_rank``, ranked over
    the hydrogen-added molecule, so a 15-heavy-atom ligand produces names like
    ``C24`` that are nowhere in the fixed ``AtomTypes`` vocabulary. Every such
    atom used to vanish before serialization, leaving a protein-only mmCIF for
    a successful ligand request.
    """
    names = ["C24", "C22", "O16", "N23"]
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    # Ligand tokens are atomized: one atom per token.
    ownership = np.array([0, 1, 2, 3])
    active = np.ones(4, dtype=bool)

    output, mask, out_names, out_elements = scatter_flat_atoms_to_folding_layout(
        positions,
        ownership,
        names,
        active,
        num_tokens=4,
        atom_elements=["C", "C", "O", "N"],
    )

    assert mask.sum() == 4
    for token, name in enumerate(names):
        slot = int(np.flatnonzero(mask[token] > 0)[0])
        assert out_names[token, slot] == name
        assert np.array_equal(output[token, slot], positions[token])
    assert sorted(out_elements[mask > 0].tolist()) == ["C", "C", "N", "O"]


def test_scatter_overflow_never_evicts_a_canonical_slot():
    """A same-row unknown name must not take a slot a known name still needs."""
    # "C1" is in the universe; "ZZ9" is not. Both belong to token 0, and the
    # unknown atom is listed first so a single-pass implementation would place
    # it at slot 0 — which is "N" — and could then collide.
    names = ["ZZ9", "N", "CA"]
    positions = np.arange(9, dtype=np.float32).reshape(3, 3)
    ownership = np.zeros(3, dtype=np.int64)
    active = np.ones(3, dtype=bool)

    output, mask, out_names, _ = scatter_flat_atoms_to_folding_layout(positions, ownership, names, active, num_tokens=1)

    assert mask.sum() == 3
    assert np.array_equal(output[0, ATOM_NAME_TO_FOLDING_INDEX["N"]], positions[1])
    assert np.array_equal(output[0, ATOM_NAME_TO_FOLDING_INDEX["CA"]], positions[2])
    overflow_slot = int(np.flatnonzero(out_names[0] == "ZZ9")[0])
    assert overflow_slot not in (ATOM_NAME_TO_FOLDING_INDEX["N"], ATOM_NAME_TO_FOLDING_INDEX["CA"])
    assert np.array_equal(output[0, overflow_slot], positions[0])


def test_scatter_raises_when_a_token_row_is_full():
    """Unrepresentable is an error — never a silent drop."""
    n = NUM_FOLDING_ATOM_TYPES + 1
    names = [f"QQ{i}" for i in range(n)]
    positions = np.zeros((n, 3), dtype=np.float32)
    ownership = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)

    with pytest.raises(ValueError, match="cannot be placed"):
        scatter_flat_atoms_to_folding_layout(positions, ownership, names, active, num_tokens=1)


def test_decode_element_symbols_offsets():
    mask = np.array([True, True, True])
    # Boltz2 convention: index == atomic number.
    boltz = np.array([6, 17, 26])
    assert decode_element_symbols(boltz, mask, index_offset=0).tolist() == ["C", "Cl", "Fe"]
    # OF3 convention: index == atomic number - 1.
    of3 = boltz - 1
    assert decode_element_symbols(of3, mask, index_offset=1).tolist() == ["C", "Cl", "Fe"]
    # Missing producer field degrades to empty strings, not a crash.
    assert decode_element_symbols(None, mask).tolist() == ["", "", ""]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("one_hot", [False, True])
def test_decode_padding(device: str, one_hot: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    chars = torch.tensor([[35, 33, 0, 55], [0, 55, 55, 55], [35, 20, 7, 0], [58, 46, 0, 0]], device=device)
    if one_hot:
        chars = torch.nn.functional.one_hot(chars, 64)
    mask = np.array([True, True, True, False])
    assert decode_atom_name_chars(chars, mask).tolist() == ["CA", "", "C4'", ""]
    assert decode_atom_name_chars(chars[:0], mask[:0]).tolist() == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("offset", [0, 1])
def test_decode_unknowns(device: str, offset: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    indices = torch.tensor([6 - offset, 17 - offset, 26 - offset, 118, 127], device=device)
    mask = np.array([True, True, False, True, True])
    encoded = torch.nn.functional.one_hot(indices, 128)
    assert decode_element_symbols(encoded, mask, index_offset=offset).tolist() == [
        "C",
        "Cl",
        "",
        "Og" if offset == 0 else "",
        "",
    ]
    assert decode_element_symbols(encoded[:0], mask[:0], index_offset=offset).tolist() == []
    # A row without a hot class is unknown, not the element at index 0.
    assert decode_element_symbols(torch.zeros_like(encoded[:1]), mask[:1], index_offset=offset).tolist() == [""]


def test_decode_strided() -> None:
    chars = np.array([[35, 33, 0, 0], [35, 20, 7, 0]])
    mask = np.ones(2, dtype=bool)
    assert decode_atom_name_chars(np.asfortranarray(chars), mask).tolist() == ["CA", "C4'"]
    assert decode_atom_name_chars(chars[::-1], mask).tolist() == ["C4'", "CA"]
