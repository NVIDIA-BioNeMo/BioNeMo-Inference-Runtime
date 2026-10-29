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

"""Text encodings for the Protenix golden fixtures; the public tree takes no binaries."""

import json
import pickle
from pathlib import Path

import torch


def dump_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Write tensors as dtype, shape and nonzero coordinates; floats round-trip exactly."""
    encoded = {}
    for name, value in tensors.items():
        dense = value.to_dense() if value.is_sparse else value
        indices = dense.nonzero(as_tuple=True)
        encoded[name] = {
            "dtype": str(dense.dtype).removeprefix("torch."),
            "shape": list(dense.shape),
            "indices": [index.tolist() for index in indices],
            "values": dense[indices].tolist(),
        }
    path.write_text(json.dumps(encoded, separators=(",", ":")) + "\n")


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    """Rebuild tensors written by :func:`dump_tensors`."""
    tensors = {}
    for name, item in json.loads(path.read_text()).items():
        dtype = getattr(torch, item["dtype"])
        value = torch.zeros(item["shape"], dtype=dtype)
        if item["values"]:
            index = tuple(torch.tensor(axis, dtype=torch.long) for axis in item["indices"])
            value[index] = torch.tensor(item["values"], dtype=torch.float64).to(dtype)
        tensors[name] = value
    return tensors


def dump_conformers(mols: dict, path: Path) -> None:
    """Write the release conformer fields that ``ComponentStore`` reads."""
    encoded = {}
    for name, mol in sorted(mols.items()):
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
        encoded[name] = {
            "atomic_numbers": [atom.GetAtomicNum() for atom in atoms],
            "formal_charges": [atom.GetFormalCharge() for atom in atoms],
            "positions": mol.GetConformer(mol.ref_conf_id).GetPositions().tolist(),
            "atom_map": dict(mol.atom_map),
            "ref_mask": [int(flag) for flag in mol.ref_mask],
        }
    path.write_text(json.dumps(encoded, indent=1, sort_keys=True) + "\n")


def build_conformer_dir(path: Path, directory: Path) -> Path:
    """Materialize :func:`dump_conformers` output as ``<component>.pkl`` RDKit molecules."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    directory.mkdir(parents=True, exist_ok=True)
    for name, item in json.loads(path.read_text()).items():
        mol = Chem.RWMol()
        for number, charge in zip(item["atomic_numbers"], item["formal_charges"], strict=True):
            atom = Chem.Atom(number)
            atom.SetFormalCharge(charge)
            mol.AddAtom(atom)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index, xyz in enumerate(item["positions"]):
            conformer.SetAtomPosition(index, Point3D(*xyz))
        mol = mol.GetMol()
        mol.ref_conf_id = mol.AddConformer(conformer, assignId=True)
        mol.atom_map = item["atom_map"]
        mol.ref_mask = item["ref_mask"]
        with (directory / f"{name}.pkl").open("wb") as stream:
            pickle.dump(mol, stream)
    return directory
