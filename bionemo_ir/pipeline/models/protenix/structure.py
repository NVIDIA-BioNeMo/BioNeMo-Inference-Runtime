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

"""Build Protenix chemical components without upstream runtime imports."""

import hashlib
import logging
import os
import pickle
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import biotite.structure as struc
import biotite.structure.info as ccd
import biotite.structure.io.pdbx as pdbx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from bionemo_ir.data.utils import MoleculeUnpickler, is_component_id, load_component_mol
from bionemo_ir.hubs.metadata import PROTENIX_MOLS_SHA256
from bionemo_ir.pipeline.utils._rng import _python_rng

from .const import DNA_CODES, PROTEIN_CODES, RNA_CODES

logger = logging.getLogger(__name__)


def _ccd_column(category: str, name: str, column: str) -> pdbx.BinaryCIFColumn:
    value = ccd.get_from_ccd(category, name, column)
    if value is None:
        raise ValueError(f"Unknown Protenix CCD component: {name}")
    return value


def conformer_dir(path: Path) -> Path:
    """Return a ``load_component_mol`` directory, splitting a release pickle once into the cache.

    The pickle is read whatever its digest, so it goes through
    ``MoleculeUnpickler``, as each component file does in ``load_component_mol``.

    Args:
        path: A directory of ``<component>.pkl`` files, or the release
            ``components.cif.rdkit_mol.pkl`` mapping of component to molecule.

    Raises:
        pickle.UnpicklingError: The pickle names any other global.
    """
    if path.is_dir():
        return path
    import bionemo_ir

    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != PROTENIX_MOLS_SHA256:
        # A truncated or foreign pickle loads cleanly but changes ligand conformers.
        logger.warning(f"Protenix conformer pickle is not the v2.0.0 release: {path} sha256={digest}")
    target = bionemo_ir.CACHE_DIR / "protenix-v2" / f"mols-{digest[:16]}"
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="bioir_mols_", dir=target.parent))
    try:
        with path.open("rb") as stream:
            for name, mol in MoleculeUnpickler(stream).load().items():
                # load_component_mol reads only these names, so others are unreachable.
                if not is_component_id(name):
                    continue
                with (staging / f"{name}.pkl").open("wb") as out:
                    pickle.dump(mol, out)
        # Concurrent workers race to publish; any complete copy serves.
        try:
            os.rename(staging, target)
        except OSError:
            if not target.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


class ComponentStore:
    """Read CCD topology from Biotite and release reference conformers per used component.

    Biotite's bundled CCD carries the release ``components.cif`` topology
    fields, so only release conformers are read, one component at a time.

    Args:
        metadata: Resolved Protenix metadata; ``ccd_mols_path`` names the
            conformer pickle or per-component directory.
    """

    def __init__(self, metadata: dict[str, str]) -> None:
        if "ccd_mols_path" not in metadata or not Path(metadata["ccd_mols_path"]).exists():
            raise FileNotFoundError("Protenix metadata missing: ccd_mols_path")
        self.mol_dir = conformer_dir(Path(metadata["ccd_mols_path"]))
        self.components: dict[str, struc.AtomArray] = {}

    def component(self, name: str) -> struc.AtomArray:
        """Return a fresh heavy-atom component, including leaving atoms."""
        if name not in self.components:
            leaving = _ccd_column("chem_comp_atom", name, "pdbx_leaving_atom_flag").as_array(str) == "Y"
            atoms = ccd.residue(name)
            atoms.set_annotation("leaving", leaving)
            atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
            if len(atoms) == 0:
                raise ValueError(f"Empty Protenix CCD component: {name}")
            mol = load_component_mol(self.mol_dir, name)
            positions = np.zeros((len(atoms), 3), dtype=np.float32)
            charge = np.zeros(len(atoms), dtype=np.int64)
            mask = np.zeros(len(atoms), dtype=np.int64)
            if mol is not None and mol.GetNumAtoms():
                indices = np.array([mol.atom_map[n] for n in atoms.atom_name])
                positions[:] = mol.GetConformer(mol.ref_conf_id).GetPositions()[indices]
                charge[:] = [mol.GetAtomWithIdx(int(i)).GetFormalCharge() for i in indices]
                mask[:] = np.asarray(mol.ref_mask)[indices]
            atoms.set_annotation("ref_pos", positions)
            atoms.set_annotation("ref_charge", charge)
            atoms.set_annotation("ref_mask", mask)
            self.components[name] = atoms
        return self.components[name].copy()

    def kind(self, name: str) -> str:
        """Polymer kind of a CCD component from ``_chem_comp.type``, as the release assigns it."""
        link = _ccd_column("chem_comp", name, "type").as_item().upper()
        if "PEPTIDE" in link and link != "PEPTIDE-LIKE":
            return "protein"
        if "DNA" in link:
            return "dna"
        if "RNA" in link:
            return "rna"
        return "ccd_ligand"

    def canonical(self, name: str, kind: str) -> str:
        """Map CCD identity onto the model's canonical residue vocabulary."""
        if kind in {"ccd_ligand", "smiles_ligand"}:
            return "UNK"
        code = _ccd_column("chem_comp", name, "one_letter_code").as_item()
        if kind == "protein":
            return PROTEIN_CODES.get(code, "UNK")
        if kind == "rna":
            return RNA_CODES.get(code, "N") if code != "I" else "N"
        return DNA_CODES.get(code, "DN") if code not in {"I", "U"} else "DN"


def polymer_atoms(store: ComponentStore, sequence: str, kind: str) -> struc.AtomArray:
    """Join standard residues and remove polymer-bond leaving groups."""
    alphabet = {"protein": PROTEIN_CODES, "rna": RNA_CODES, "dna": DNA_CODES}[kind]
    parts = []
    for index, letter in enumerate(sequence):
        if letter not in alphabet:
            raise ValueError(f"Invalid {kind} residue: {letter}")
        atoms = store.component(alphabet[letter])
        atoms.res_id[:] = index + 1
        parts.append(atoms)
    if not parts:
        raise ValueError("Empty Protenix polymer sequence")
    for index, atoms in enumerate(parts):
        # Leaving oxygens depend on terminal position.
        removed = set()
        if kind == "protein" and index < len(parts) - 1:
            removed.add("OXT")
        if kind in {"rna", "dna"} and index > 0:
            removed.add("OP3")
        parts[index] = atoms[~np.isin(atoms.atom_name, list(removed))]
    result = struc.concatenate(parts)
    if result.bonds is None:
        result.bonds = struc.BondList(len(result))
    offsets = np.cumsum([0] + [len(p) for p in parts])
    outgoing, incoming = ("C", "N") if kind == "protein" else ("O3'", "P")
    for index in range(len(parts) - 1):
        left = np.flatnonzero(parts[index].atom_name == outgoing)
        right = np.flatnonzero(parts[index + 1].atom_name == incoming)
        if left.size != 1 or right.size != 1:
            raise ValueError("Protenix polymer connection atoms missing")
        result.bonds.add_bond(int(offsets[index] + left[0]), int(offsets[index + 1] + right[0]), 1)
    return result


def smiles_atoms(sequence: str) -> struc.AtomArray:
    """Build a seeded SMILES conformer and retain heavy-atom ordering."""
    molecule = Chem.MolFromSmiles(sequence)
    if molecule is None:
        raise ValueError("Invalid Protenix ligand SMILES")
    molecule = Chem.AddHs(molecule)
    seed = _python_rng().randrange(1, 2**31)
    status = AllChem.EmbedMolecule(molecule, randomSeed=seed)
    if status:
        status = AllChem.EmbedMolecule(molecule, randomSeed=seed, useRandomCoords=True)
    if status:
        raise ValueError("Protenix ligand conformer generation failed")
    atoms = struc.AtomArray(molecule.GetNumAtoms())
    atoms.res_name[:] = "UNL"
    atoms.res_id[:] = 1
    atoms.hetero[:] = True
    counts = Counter()
    names, elements, charges = [], [], []
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol().upper()
        counts[symbol] += 1
        names.append(f"{symbol}{counts[symbol]}")
        elements.append(symbol)
        charges.append(atom.GetFormalCharge())
    atoms.atom_name = np.asarray(names)
    atoms.element = np.asarray(elements)
    atoms.coord[:] = molecule.GetConformer().GetPositions()
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 1] for b in molecule.GetBonds()]
    atoms.bonds = struc.BondList(len(atoms), np.asarray(bonds, dtype=np.uint32).reshape(-1, 3))
    atoms.set_annotation("ref_pos", atoms.coord.copy())
    atoms.set_annotation("ref_charge", np.asarray(charges, dtype=np.int64))
    atoms.set_annotation("ref_mask", np.ones(len(atoms), dtype=np.int64))
    atoms.set_annotation("leaving", np.zeros(len(atoms), dtype=bool))
    return atoms[atoms.element != "H"]
