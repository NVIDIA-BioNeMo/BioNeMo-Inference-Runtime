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

"""Protenix token, atom, bond, and reference-frame features."""

import biotite.structure as struc
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from bionemo_ir.pipeline.base import FeatureGeneratorBase
from bionemo_ir.pipeline.models.boltz2.const import res_to_disto_atom
from bionemo_ir.pipeline.utils.atom import encode_atom_name_chars_one_hot

from .const import MOL_TYPES, RESTYPES

_ELEMENT_INDEX = {Chem.GetPeriodicTable().GetElementSymbol(z).upper(): z - 1 for z in range(1, 119)}


def structure_context(context: dict) -> dict:
    return context["_row"]["structure"]


def token_restypes(structure: dict) -> torch.Tensor:
    """Canonical residue id per token, shaped ``[N_token]``.

    Args:
        structure: Tokenized structure context; reads ``tokens[*]["center"]``
            and ``atoms.canonical``.
    """
    centers = [t["center"] for t in structure["tokens"]]
    return torch.tensor([RESTYPES[n] for n in structure["atoms"].canonical[centers]])


class TokenFeatureGenerator(FeatureGeneratorBase):
    """Encode token identity and model confidence-atom indices."""

    def __call__(self, batch: dict, context: dict) -> dict:
        structure = structure_context(context)
        atoms, tokens = structure["atoms"], structure["tokens"]
        centers = np.asarray([t["center"] for t in tokens])
        token_atoms = atoms[centers]
        features = {
            "token_index": torch.arange(len(tokens)),
            "residue_index": torch.as_tensor(token_atoms.res_id.astype(np.int64)),
            "asym_id": torch.as_tensor(token_atoms.asym),
            "entity_id": torch.as_tensor(token_atoms.entity),
            "sym_id": torch.as_tensor(token_atoms.sym),
            "restype": F.one_hot(token_restypes(structure), 32).float(),
            "atom_to_token_idx": torch.as_tensor(atoms.atom_to_token),
            "atom_to_tokatom_idx": torch.as_tensor(atoms.local_atom),
            "token_mask": torch.ones(len(tokens)),
            "atom_mask": torch.ones(len(atoms)),
        }
        features["is_ligand"] = torch.as_tensor((atoms.mol_type == MOL_TYPES["ccd_ligand"]).astype(np.int64))
        return features


class BondFeatureGenerator(FeatureGeneratorBase):
    """Encode token connectivity and representative-atom masks."""

    def __call__(self, batch: dict, context: dict) -> dict:
        structure = structure_context(context)
        atoms, tokens = structure["atoms"], structure["tokens"]
        nt, na = len(tokens), len(atoms)
        distogram = torch.zeros(na, dtype=torch.long)
        for token in tokens:
            indices, center = token["atoms"], token["center"]
            if not token["standard"]:
                distogram[center] = 1
                continue
            representative = res_to_disto_atom.get(str(atoms.res_name[center]), "CB")
            matches = [i for i in indices if atoms.atom_name[i] == representative]
            distogram[matches[0] if matches else indices[0]] = 1
        standard = np.array([t["standard"] for t in tokens], dtype=bool)
        left, right = atoms.bonds.as_array()[:, :2].astype(np.int64).T
        lt, rt = atoms.atom_to_token[left], atoms.atom_to_token[right]
        ligand_code = MOL_TYPES["ccd_ligand"]
        ligand = (atoms.mol_type[left] == ligand_code) | (atoms.mol_type[right] == ligand_code)
        same_group = ~standard[lt] & ~standard[rt] & (atoms.ref_space_uid[left] == atoms.ref_space_uid[right])
        bonded = ligand | same_group
        token_bonds = torch.zeros(nt, nt)
        token_bonds[lt[bonded], rt[bonded]] = 1
        token_bonds[rt[bonded], lt[bonded]] = 1
        return {"token_bonds": token_bonds, "distogram_rep_atom_mask": distogram}


def token_frames(atoms: struc.AtomArray, tokens: list[dict], positions: np.ndarray) -> torch.Tensor:
    """Flag tokens with a valid polymer backbone or nearest-neighbor ligand frame."""
    valid = []
    trees = {}
    for token in tokens:
        indices, center = token["atoms"], token["center"]
        if token["standard"] and len(indices) > 1:
            labels = ["N", "CA", "C"] if atoms.mol_type[center] == 0 else ["C1'", "C3'", "C4'"]
            lookup = {atoms.atom_name[i]: i for i in indices}
            frame = [lookup.get(n, -1) for n in labels]
            keep = -1 not in frame
        else:
            uid = atoms.ref_space_uid[center]
            if uid not in trees:
                members = np.flatnonzero(atoms.ref_space_uid == uid)
                trees[uid] = (members, KDTree(positions[members]) if len(members) >= 3 else None)
            members, tree = trees[uid]
            frame, keep = [-1, center, -1], False
            if tree is not None:
                _, closest = tree.query(positions[center], k=3)
                frame = [int(members[closest[1]]), center, int(members[closest[2]])]
                a, b, c = positions[frame]
                v, w = b - a, c - b
                nv, nw = np.linalg.norm(v), np.linalg.norm(w)
                angle = np.degrees(np.arccos(np.clip(np.dot(v, w) / (nv * nw + 1e-4), -1, 1)))
                keep = bool(
                    atoms.ref_mask[frame].all() and not np.isclose(nv, 0) and not np.isclose(nw, 0) and 25 < angle < 155
                )
        valid.append(int(keep))
    return torch.tensor(valid)


class ReferenceFeatureGenerator(FeatureGeneratorBase):
    """Augment cached reference conformers with request-local randomness."""

    def __call__(self, batch: dict, context: dict) -> dict:
        structure = structure_context(context)
        atoms = structure["atoms"]
        rng = context["_rng"].numpy
        positions = np.empty_like(atoms.ref_pos)
        # Reference groups are contiguous residues, numbered in order.
        bounds = np.flatnonzero(np.diff(atoms.ref_space_uid)) + 1
        for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(atoms)], strict=True):
            centered = atoms.ref_pos[start:stop] - atoms.ref_pos[start:stop].mean(0)
            translation = rng.uniform(-1, 1, 3)
            rotation = Rotation.random(random_state=rng).as_matrix()
            positions[start:stop] = (centered + translation) @ rotation.T
        frames = token_frames(atoms, structure["tokens"], positions)
        return {
            "ref_pos": torch.as_tensor(positions, dtype=torch.float32),
            "ref_mask": torch.as_tensor(atoms.ref_mask),
            "ref_charge": torch.as_tensor(atoms.ref_charge),
            "ref_element": F.one_hot(torch.tensor([_ELEMENT_INDEX[e] for e in atoms.element]), 128),
            "ref_atom_name_chars": encode_atom_name_chars_one_hot(atoms.atom_name).long(),
            "ref_space_uid": torch.as_tensor(atoms.ref_space_uid),
            "has_frame": frames,
        }
