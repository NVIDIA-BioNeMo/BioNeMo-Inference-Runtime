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
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .schemas import AtomType, AtomTypes, ResType, ResTypes

logger = logging.getLogger(__name__)


@lru_cache
def get_all_residue_types(model: str, include_gap: bool = True) -> list[ResType]:
    """Return the ordered residue-type list for a given model.

    ``include_gap`` is only honored for the openfold2/alphafold2 branch. The
    boltz and openfold3 branches always include GAP because their feature
    schemas require the full restype vocabulary (gap is at a fixed index).
    """
    if "openfold2" in model or "alphafold2" in model:
        ret = ResTypes.basic_20_residue_types() + [ResTypes.X]
        if include_gap:
            ret.append(ResTypes.GAP)
        return ret
    elif "boltz" in model:
        # Order must match boltz2's ``tokens`` table in
        # pipeline/models/boltz2/const.py (33 entries):
        #   idx 23-26: A, G, C, U  (biological purine-first order, NOT alphabetical)
        #   idx 28-31: DA, DG, DC, DT
        # The model emits residue_type indices against THIS table; if the
        # writer's res_types list is alphabetical (RA, RC, RG, RU) instead,
        # the CIF writer's entity_seq lookup swaps C↔G and breaks lDDT
        # scoring on RNA/DNA chains. Do NOT use rna_nucleotide_types() /
        # dna_nucleotide_types() here — their alphabetical order is wrong.
        return (
            [ResTypes.PAD, ResTypes.GAP]
            + ResTypes.basic_20_residue_types()
            + [ResTypes.X]
            + [ResTypes.RA, ResTypes.RG, ResTypes.RC, ResTypes.RU]
            + [ResTypes.RX]
            + [ResTypes.DA, ResTypes.DG, ResTypes.DC, ResTypes.DT]
            + [ResTypes.DX]
        )
    elif "openfold3" in model:
        # Order must match RESTYPES_3 in pipeline/models/openfold3/const.py (32 types):
        #   idx 21-25: A, G, C, U, N   idx 26-30: DA, DG, DC, DT, DN
        # ResTypes has no RN (RNA any-nucleotide) or DN (DNA any-nucleotide) entries;
        # RX and DX (unknown) are used as stand-ins for indices 25 and 30.
        # Do NOT use rna_nucleotide_types() / dna_nucleotide_types() here — their
        # alphabetical order (A,C,G,U / DA,DC,DG,DT) does not match RESTYPES_3.
        return ResTypes.basic_20_residue_types() + [
            ResTypes.X,
            ResTypes.RA,
            ResTypes.RG,
            ResTypes.RC,
            ResTypes.RU,
            ResTypes.RX,  # idx 21-25
            ResTypes.DA,
            ResTypes.DG,
            ResTypes.DC,
            ResTypes.DT,
            ResTypes.DX,  # idx 26-30
            ResTypes.GAP,
        ]
    else:
        raise ValueError(f"Invalid model: {model}")


@lru_cache
def get_all_atom_types(model: str) -> list[AtomType]:
    """Return all atom types for a given model."""
    atom_types = AtomTypes.all_types()
    return atom_types


def sequence_to_onehot(sequence: str, restype_to_idx: dict[str, int]) -> torch.IntTensor:
    """
    Maps the given sequence into a one-hot encoded matrix.
    """
    indices = []
    for residue in sequence:
        indices.append(restype_to_idx[residue])

    indices = torch.tensor(indices, dtype=torch.long)
    return F.one_hot(indices, num_classes=len(restype_to_idx))


# ---------------------------------------------------------------------------
# CCD-component mol loading — model-agnostic (Boltz2, OF3, Protenix).
# ---------------------------------------------------------------------------
def load_component_mol(mol_dir: str | Path, name: str) -> Any | None:
    """Load one CCD-component RDKit ``Mol`` from ``<mol_dir>/<name>.pkl``.

    ``name`` is a residue/component id that may originate from an untrusted
    input CIF and is used to build a path that is ``pickle.load``-ed, so it is
    validated as a bare CCD-style identifier (ASCII alphanumeric, <=5 chars —
    CCD ids are at most 5 chars) and the resolved path is confirmed to stay
    under ``mol_dir``. Either check failing, or the pickle being absent, returns
    ``None`` (callers fall back to their unknown-component path).
    """
    if not (name.isascii() and name.isalnum() and len(name) <= 5):
        return None
    from rdkit import Chem

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    base = Path(mol_dir).resolve()
    p = (base / f"{name}.pkl").resolve()
    if base not in p.parents:  # defense-in-depth: stay under mol_dir
        return None
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# gemmi structure read + normalize — model-agnostic template helpers.
# ---------------------------------------------------------------------------
def read_gemmi_structure(source: str, fmt: str, from_content: bool):
    """Read an mmCIF/PDB path or raw content string into a gemmi ``Structure``.

    ``fmt`` is ``"cif"``/``"mmcif"`` or ``"pdb"``; ``from_content`` selects raw
    text vs a filesystem path.
    """
    import gemmi

    fmt = fmt.lower()
    if fmt in ("mmcif", "cif"):
        doc = gemmi.cif.read_string(source) if from_content else gemmi.cif.read(str(source))
        return gemmi.make_structure_from_block(doc[0])
    if fmt == "pdb":
        return gemmi.read_pdb_string(source) if from_content else gemmi.read_pdb(str(source))
    raise ValueError(f"Unsupported template format: {fmt!r}")


def normalize_gemmi_structure(st) -> None:
    """Clean + entity-normalize a gemmi structure in place (mirrors OSS parse_mmcif).

    Removes waters/hydrogens/altconfs/empty chains, then synthesizes an entity
    sequence from observed residues ONLY when gemmi has none — keeping any
    existing SEQRES, since overwriting it renumbers res_idx and shifts the
    alignment offset (off-by-N).
    """
    st.merge_chain_parts()
    st.remove_waters()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    st.remove_empty_chains()
    st.setup_entities()
    for chain in st[0]:
        poly = chain.get_polymer()
        if len(poly) == 0:
            continue
        ent = st.get_entity_of(poly)
        if not ent.full_sequence:
            ent.full_sequence = [res.name for res in poly]
