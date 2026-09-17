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
"""``get_token_frame_mask``: which tokens may be the aligned token of pTM / ipTM.

The mask is the ``has_frame`` input of the pTM / ipTM outer maximum (AF3 SI
5.9.1), so a token wrongly excluded silently lowers the reported score and a
token wrongly included raises it. Upstream builds it from the sampled
coordinates (``openfold3/core/metrics/aggregate_confidence_ranking.py``), which
makes three behaviours load-bearing and asserted here:

  * a polymer residue is eligible when its backbone frame atoms are present,
    and not when one of them is missing;
  * an **atomized** token (ligand atom) is eligible when its two nearest
    same-chain neighbours span a sane angle -- ligands are not excluded as a
    class, which is what distinguishes the real mask from a ``~is_atomized``
    approximation;
  * a token with no usable in-chain neighbours (a lone ion) and a token whose
    neighbours are collinear are both ineligible.

The mask depends on the coordinates, so it carries the diffusion-sample axis.
"""

import math

import pytest
import torch

from bionemo_ir._torch.modules.openfold3.utils.atomize_utils import get_token_frame_mask
from bionemo_ir._torch.modules.openfold3.utils.token_atom_constants import (
    TOKEN_NAME_TO_ATOM_NAMES,
    TOKEN_TYPES_WITH_GAP,
)

# An amino acid whose frame atoms (N, CA, C) are the first three of its atom list.
_RESIDUE = "ALA"
_RESIDUE_ATOMS = TOKEN_NAME_TO_ATOM_NAMES[_RESIDUE]
_GAP = TOKEN_TYPES_WITH_GAP.index("GAP")


def _build(tokens: list[tuple[int, str | None]], positions, *, missing_atoms=()):
    """Assemble an OF3 batch and coordinates from a token spec.

    Args:
        tokens: one ``(chain_id, restype)`` per token. ``restype=None`` marks an
            atomized token (a ligand or ion atom), which owns exactly one atom.
        positions: ``[..., N_atom, 3]`` atom coordinates.
        missing_atoms: flat atom indices to mark absent in ``atom_mask``.

    Returns:
        ``(batch, x)`` ready for :func:`get_token_frame_mask`.
    """
    n_atoms = [1 if restype is None else len(_RESIDUE_ATOMS) for _, restype in tokens]
    chains = [chain for chain, _ in tokens]
    atomized = [restype is None for _, restype in tokens]
    n_token = len(tokens)

    num_atoms_per_token = torch.tensor(n_atoms, dtype=torch.long)[None]
    start_atom_index = torch.cat([torch.zeros(1, dtype=torch.long), num_atoms_per_token[0].cumsum(0)[:-1]])[None]
    n_atom = int(num_atoms_per_token.sum())

    restype = torch.zeros(1, n_token, len(TOKEN_TYPES_WITH_GAP))
    for i, (_, name) in enumerate(tokens):
        restype[0, i, _GAP if name is None else TOKEN_TYPES_WITH_GAP.index(name)] = 1.0

    atom_mask = torch.ones(1, n_atom)
    atom_mask[0, list(missing_atoms)] = 0.0

    is_atomized = torch.tensor(atomized, dtype=torch.int32)[None]
    batch = {
        "token_mask": torch.ones(1, n_token),
        "num_atoms_per_token": num_atoms_per_token,
        "asym_id": torch.tensor(chains, dtype=torch.long)[None],
        "start_atom_index": start_atom_index,
        "is_protein": (1 - is_atomized).float(),
        "is_rna": torch.zeros(1, n_token),
        "is_dna": torch.zeros(1, n_token),
        "is_atomized": is_atomized,
        "restype": restype,
        "atom_mask": atom_mask,
    }
    x = torch.as_tensor(positions, dtype=torch.float32)
    assert x.shape[-2] == n_atom, (x.shape, n_atom)
    return batch, x


def _residue_positions(origin, count):
    """``count`` residues of spread-out atoms, offset along x from ``origin``."""
    out = []
    for r in range(count):
        for a in range(len(_RESIDUE_ATOMS)):
            out.append([origin[0] + 4.0 * r + 0.6 * a, origin[1] + 0.4 * a, origin[2]])
    return out


def _tetrahedral_ligand(origin, count):
    """``count`` ligand atoms on a zig-zag, so neighbour angles sit near 109 degrees."""
    out = []
    for i in range(count):
        out.append([origin[0] + 1.4 * i, origin[1] + (0.9 if i % 2 else 0.0), origin[2]])
    return out


def test_polymer_residue_is_eligible_and_loses_its_frame_with_a_missing_backbone_atom():
    """A standard residue is eligible; removing one frame atom makes it ineligible."""
    tokens = [(1, _RESIDUE)] * 3
    positions = _residue_positions((0.0, 0.0, 0.0), 3)

    batch, x = _build(tokens, [positions])
    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()
    assert mask.tolist() == [[True, True, True]]

    # Drop the N of the middle residue: N/CA/C are the first three atoms of ALA.
    n_of_second_residue = len(_RESIDUE_ATOMS)
    batch, x = _build(tokens, [positions], missing_atoms=(n_of_second_residue,))
    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()
    assert mask.tolist() == [[True, False, True]]


def test_ligand_atoms_are_eligible_when_their_neighbour_angle_is_sane():
    """Atomized tokens are admitted on geometry, not excluded as a class."""
    n_residues, n_ligand = 4, 6
    tokens = [(1, _RESIDUE)] * n_residues + [(2, None)] * n_ligand
    positions = _residue_positions((0.0, 0.0, 0.0), n_residues) + _tetrahedral_ligand((40.0, 0.0, 0.0), n_ligand)

    batch, x = _build(tokens, [positions])
    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()[0]

    assert mask[:n_residues].all()
    # The interior ligand atoms have two non-collinear neighbours and qualify; a
    # ``~is_atomized`` mask would have excluded every one of them.
    assert mask[n_residues:].any()
    assert int(mask[n_residues:].sum()) >= n_ligand - 2


def test_lone_ion_and_collinear_ligand_have_no_valid_frame():
    """No in-chain neighbours, or collinear ones, means no frame."""
    n_residues = 3
    # A single-atom chain: its two nearest in-chain candidates do not exist.
    tokens = [(1, _RESIDUE)] * n_residues + [(2, None)]
    positions = _residue_positions((0.0, 0.0, 0.0), n_residues) + [[40.0, 0.0, 0.0]]
    batch, x = _build(tokens, [positions])
    assert not get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()[0, -1]

    # A straight chain of ligand atoms: the angle at an interior atom is 180 degrees.
    n_ligand = 5
    tokens = [(1, _RESIDUE)] * n_residues + [(2, None)] * n_ligand
    positions = _residue_positions((0.0, 0.0, 0.0), n_residues) + [[40.0 + 1.4 * i, 0.0, 0.0] for i in range(n_ligand)]
    batch, x = _build(tokens, [positions])
    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()[0]
    assert mask[:n_residues].all()
    assert not mask[n_residues:].any()


@pytest.mark.parametrize("n_atoms", [1, 2])
def test_tiny_atomized_inputs_have_no_valid_frame(n_atoms):
    """Fewer than three atoms cannot form a frame."""
    tokens = [(1, None)] * n_atoms
    batch, x = _build(tokens, [_tetrahedral_ligand((0.0, 0.0, 0.0), n_atoms)])
    x = x.unsqueeze(1).expand(1, 3, *x.shape[1:])

    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()

    assert mask.shape == (1, 3, n_atoms)
    assert not mask.any()


def test_mask_is_per_diffusion_sample():
    """The mask follows the sampled coordinates, so it carries the sample axis."""
    n_residues, n_ligand = 3, 5
    tokens = [(1, _RESIDUE)] * n_residues + [(2, None)] * n_ligand
    polymer = _residue_positions((0.0, 0.0, 0.0), n_residues)
    bent = polymer + _tetrahedral_ligand((40.0, 0.0, 0.0), n_ligand)
    straight = polymer + [[40.0 + 1.4 * i, 0.0, 0.0] for i in range(n_ligand)]

    batch, x = _build(tokens, [[bent, straight]])
    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()

    assert mask.shape == (1, 2, len(tokens))
    assert mask[0, 0, n_residues:].any(), "bent ligand: some atoms are frame-eligible"
    assert not mask[0, 1, n_residues:].any(), "straight ligand: none are"
    # Polymer tokens do not depend on the ligand geometry.
    assert mask[0, :, :n_residues].all()


def test_batch_features_broadcast_across_diffusion_samples():
    """Batch-shaped features gain a sample axis without mixing batch entries."""
    tokens = [(1, None)] * 5
    positions = _tetrahedral_ligand((0.0, 0.0, 0.0), len(tokens))
    batch, x = _build(tokens, [positions])

    batch_size, n_samples = 3, 2
    batch = {key: value.expand(batch_size, *value.shape[1:]).clone() for key, value in batch.items()}
    batch["asym_id"][1] = torch.arange(1, len(tokens) + 1)
    batch["atom_mask"][2] = 0.0
    x = x.unsqueeze(1).expand(batch_size, n_samples, *x.shape[1:])

    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"]).bool()

    assert mask.shape == (batch_size, n_samples, len(tokens))
    assert mask[0].any()
    assert not mask[1].any()
    assert not mask[2].any()


@pytest.mark.parametrize("angle_threshold", [25.0, 60.0])
def test_angle_threshold_bounds_the_admitted_ligand_geometry(angle_threshold):
    """A ligand angle inside the threshold of 0 degrees is rejected at both settings."""
    n_residues = 3
    tokens = [(1, _RESIDUE)] * n_residues + [(2, None)] * 3
    # Two neighbours almost on top of each other, seen from the middle atom:
    # the angle at the query atom is ~10 degrees, inside either threshold.
    narrow = math.radians(10.0)
    positions = _residue_positions((0.0, 0.0, 0.0), n_residues) + [
        [40.0, 0.0, 0.0],
        [40.0 + 3.0, 0.0, 0.0],
        [40.0 + 3.0 * math.cos(narrow), 3.0 * math.sin(narrow), 0.0],
    ]
    batch, x = _build(tokens, [positions])
    mask = get_token_frame_mask(batch=batch, x=x, atom_mask=batch["atom_mask"], angle_threshold=angle_threshold).bool()[
        0
    ]
    assert mask[:n_residues].all()
    assert not mask[n_residues]
