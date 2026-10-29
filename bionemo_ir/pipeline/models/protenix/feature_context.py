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

"""Translate parsed requests into Protenix atom and token context."""

import biotite.structure as struc
import numpy as np

from bionemo_ir.configs import BaseConfig
from bionemo_ir.pipeline.base import ContextGeneratorBase
from bionemo_ir.pipeline.models.boltz2.const import res_to_center_atom

from .const import MOL_TYPES, RES_ATOMS_DICT, RESTYPES
from .structure import ComponentStore, polymer_atoms, smiles_atoms


class ProtenixContextGenerator(ContextGeneratorBase):
    """Construct chemical identity before numerical featurization."""

    def __init__(self, config: BaseConfig | None = None, metadata: dict | None = None) -> None:
        super().__init__(config, metadata)
        self.store: ComponentStore | None = None

    def prepare(self) -> None:
        # Each worker loads local metadata; never ship the CCD through Ray.
        if self.store is None:
            self.store = ComponentStore(self.metadata or {})

    def __call__(self, parsed: dict) -> dict:
        self.prepare()
        polymers = parsed.get("polymers")
        if not polymers:
            raise ValueError("No Protenix polymers supplied")
        chains, parts, used_ids = [], [], set()
        # Upstream sorts string-valued entity identifiers.
        entity_ids = {int(v) - 1: i for i, v in enumerate(sorted(str(i + 1) for i in range(len(polymers))))}
        for entity, polymer in enumerate(polymers):
            kind, sequence = polymer["polymer_type"], polymer["sequence"]
            if kind not in MOL_TYPES:
                raise ValueError(f"Unsupported Protenix polymer: {kind}")
            if kind in {"protein", "rna", "dna"}:
                entity_atoms = polymer_atoms(self.store, sequence, kind)
            elif kind == "smiles_ligand":
                entity_atoms = smiles_atoms(sequence)
            else:
                components = []
                for residue, name in enumerate(sequence.split("_"), 1):
                    atoms = self.store.component(name)
                    atoms.res_id[:] = residue
                    # A peptide or nucleotide CCD ligand (e.g. TYR) is a standard residue upstream.
                    atoms.set_annotation("mol_type", np.full(len(atoms), MOL_TYPES[self.store.kind(name)]))
                    components.append(atoms)
                entity_atoms = struc.concatenate(components)
            if kind != "ccd_ligand":
                entity_atoms.set_annotation("mol_type", np.full(len(entity_atoms), MOL_TYPES[kind]))
            ids = polymer["chain_id"]
            ids = [ids] if isinstance(ids, str) else ids
            for copy, chain_id in enumerate(ids):
                if chain_id in used_ids:
                    raise ValueError(f"Duplicate Protenix chain: {chain_id}")
                used_ids.add(chain_id)
                atoms = entity_atoms.copy()
                atoms.set_annotation("chain_id", np.full(len(atoms), chain_id))
                atoms.set_annotation("asym", np.full(len(atoms), len(chains), dtype=np.int64))
                atoms.set_annotation("entity", np.full(len(atoms), entity_ids[entity], dtype=np.int64))
                atoms.set_annotation("sym", np.full(len(atoms), copy, dtype=np.int64))
                atoms.hetero[:] = kind in {"ccd_ligand", "smiles_ligand"}
                chains.append({**polymer, "chain_id": chain_id, "entity_id": entity_ids[entity]})
                parts.append(atoms)
        atoms = struc.concatenate(parts)
        starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
        tokens, atom_to_token, local_atom, groups, canonical = [], [], [], [], []
        for uid, (start, stop) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            name = str(atoms.res_name[start])
            kind = chains[int(atoms.asym[start])]["polymer_type"]
            standard = name in RESTYPES and atoms.mol_type[start] != MOL_TYPES["ccd_ligand"]
            if kind == "ccd_ligand":
                kind = self.store.kind(name)
            residue_type = self.store.canonical(name, kind) if name != "UNL" else "UNK"
            ranges = [(start, stop)] if standard else [(i, i + 1) for i in range(start, stop)]
            for left, right in ranges:
                indices = list(range(left, right))
                if standard:
                    center = res_to_center_atom[name]
                    candidates = [i for i in indices if atoms.atom_name[i] == center]
                    if len(candidates) != 1:
                        raise ValueError(f"Protenix token center missing: {name}")
                    center_index = candidates[0]
                else:
                    center_index = left
                tokens.append({"atoms": indices, "center": center_index, "standard": standard})
                atom_to_token.extend([len(tokens) - 1] * (right - left))
            local_atom.extend([RES_ATOMS_DICT[name][n] if standard else 0 for n in atoms.atom_name[start:stop]])
            groups.extend([uid] * (stop - start))
            canonical.extend([residue_type] * (stop - start))
        atoms.set_annotation("canonical", np.asarray(canonical))
        atoms.set_annotation("ref_space_uid", np.asarray(groups, dtype=np.int64))
        atoms.set_annotation("atom_to_token", np.asarray(atom_to_token, dtype=np.int64))
        atoms.set_annotation("local_atom", np.asarray(local_atom, dtype=np.int64))
        return {"structure": {"atoms": atoms, "tokens": tokens, "chains": chains}}
