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

"""Boltz2 ContextGenerator.

Builds the per-row context for the downstream feature pipeline:

* Constructs a :class:`Structure` from :class:`InputParsed` (protein, RNA, DNA,
  CCD ligand, SMILES ligand).
* Tokenizes the structure to produce :class:`Token` / :class:`TokenBond` lists.
* Loads the per-residue RDKit molecules required for atom-level features. CCD
  components are loaded from ``mol_dir``; SMILES ligands are generated on the
  fly inside :mod:`structure` and injected directly into the molecules dict.
* Parses one MSA entry per chain — non-protein chains get an empty/None entry
  which the MSA featurizer interprets as single-sequence mode.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from tensorrt_bionemo.data.schemas.basic import InputParsed
from tensorrt_bionemo.data.utils import load_component_mol
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase

# isort: off
from .const import canonical_tokens
from .structure import build_structure_from_input
from .tokenizer_logic import tokenize_structure
# isort: on


def _load_ccd(ccd_path: str | Path) -> dict[str, Any]:
    with open(ccd_path, "rb") as f:
        return pickle.load(f)


def _load_molecules(mol_dir: str | Path, names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        mol = load_component_mol(mol_dir, name)
        if mol is not None:
            out[name] = mol
    return out


class Boltz2ContextGenerator(ContextGeneratorBase):
    """Generate Boltz2 feature dict from InputParsed using CCD and mols/.

    CCD and molecules are loaded lazily on first call and cached for reuse.
    SMILES-derived ligand molecules are generated per-call (they are
    request-dependent) and merged into the returned molecules map.
    """

    def __init__(
        self,
        config: Any | None = None,
        metadata: dict[str, Any] | None = None,
        ccd_path: str | Path | None = None,
        mol_dir: str | Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(config)
        meta = metadata or {}
        ccd = ccd_path or meta.get("ccd_path")
        mol = mol_dir or meta.get("mol_dir")
        self._ccd_path = Path(ccd) if ccd else None
        self._mol_dir = Path(mol) if mol else None
        self._required_kwargs = ["parsed"]
        self._ccd: dict[str, Any] | None = None
        self._molecules_cache: dict[str, Any] = {}
        self._molecules_binary_cache: dict[str, bytes] = {}

    @property
    def required_kwargs(self) -> list[str]:
        return self._required_kwargs

    @required_kwargs.setter
    def required_kwargs(self, value: list[str]) -> None:
        self._required_kwargs = value

    def _get_ccd(self) -> dict[str, Any]:
        if self._ccd is None:
            self._ccd = _load_ccd(self._ccd_path)
        return self._ccd

    def _get_molecules(self, names: set[str]) -> dict[str, bytes]:
        """Return binary-serialized molecules, loading only uncached ones from mol_dir."""
        missing = names - self._molecules_cache.keys()
        if missing:
            loaded = _load_molecules(self._mol_dir, list(missing))
            self._molecules_cache.update(loaded)
            for k, v in loaded.items():
                self._molecules_binary_cache[k] = self._serialize_mol(v)
        return {k: self._molecules_binary_cache[k] for k in names if k in self._molecules_binary_cache}

    @staticmethod
    def _serialize_mol(mol) -> bytes:
        return mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        if not parsed.get("polymers"):
            raise ValueError("No polymers in input")
        if self._ccd_path is None or not self._ccd_path.exists():
            raise ValueError("ccd_path must be set and exist")
        if self._mol_dir is None or not self._mol_dir.is_dir():
            raise ValueError("mol_dir must be set and exist")

        ccd = self._get_ccd()
        structure, extra_mols, constraints = build_structure_from_input(parsed, ccd)
        tokens, token_bonds = tokenize_structure(structure)

        # CCD-backed mol names (canonical tokens + per-residue ligand codes).
        mol_names = set(canonical_tokens)
        extra_names = set(extra_mols.keys())
        for t in tokens:
            if t.res_name not in extra_names:
                mol_names.add(t.res_name)
        molecules_bin = self._get_molecules(mol_names)
        missing = [n for n in mol_names if n not in molecules_bin]
        if missing:
            raise ValueError(f"Missing molecules in mol_dir: {missing}")

        # Merge SMILES-derived mols (serialised same way as disk-loaded ones).
        for name, mol in extra_mols.items():
            molecules_bin[name] = self._serialize_mol(mol)

        msa_per_chain = []
        paired_msa_per_chain = []
        # Iterate in the SAME entity-grouped order build_structure_from_input uses
        # (it set ``_entity_id`` on each polymer above), so these per-chain MSA
        # lists line up positionally with the reordered chains that
        # process_msa_features indexes — otherwise interleaved homo-oligomers
        # (e.g. 1a3n A,C,B,D) would pick up the wrong MSA rows.
        for poly in sorted(parsed.get("polymers") or [], key=lambda p: p.get("_entity_id", 0)):
            chain_ids = poly.get("chain_id") or ["_"]
            n_chains_in_poly = len(chain_ids) if isinstance(chain_ids, list) else 1
            ptype = (poly.get("polymer_type") or "protein").lower()
            # Only protein chains carry MSAs in Boltz2; for nucleic acids /
            # ligands fall through to the gap-only "single-sequence" path.
            if ptype == "protein":
                msa_entry = self._load_msa_entry(poly.get("msas"))
                paired_entry = self._load_msa_entry(poly.get("paired_msas"))
            else:
                msa_entry = None
                paired_entry = None
            for _ in range(n_chains_in_poly):
                msa_per_chain.append(msa_entry)
                paired_msa_per_chain.append(paired_entry)
        if not msa_per_chain:
            msa_per_chain = [None]
            paired_msa_per_chain = [None]

        # Thread structural templates into the row for the template feature
        # generator. One entry per PROTEIN polymer that carries templates
        # (templates are protein-only), giving the query chain names, the query
        # sequence, and the polymer's list of TemplateParsed (content/format/
        # chain_id). If no polymer has templates, the key is omitted so the
        # generator falls through to the byte-identical dummy path.
        templates_row = []
        for poly in parsed.get("polymers") or []:
            ptype = (poly.get("polymer_type") or "protein").lower()
            if ptype != "protein":
                continue
            tmpls = poly.get("templates")
            if not tmpls:
                continue
            chain_ids = poly.get("chain_id")
            if chain_ids is None:
                chain_ids = ["A"]  # matches build_structure_from_input default
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            templates_row.append(
                {
                    "chain_ids": list(chain_ids),
                    "sequence": poly.get("sequence") or "",
                    "templates": list(tmpls),
                }
            )

        row = {
            "structure": structure.to_dict(),
            "tokens": [t.to_dict() for t in tokens],
            "token_bonds": [tb.to_dict() for tb in token_bonds],
            "molecules": molecules_bin,
            "msa_parsed_per_chain": msa_per_chain,
            "paired_msa_per_chain": paired_msa_per_chain,
            "residue_constraints": constraints,
        }
        if templates_row:
            row["templates"] = templates_row
            # mol_dir lets the template generator load CCD components for
            # modified template residues (mirrors OSS parse_ccd_residue).
            if self._mol_dir is not None:
                row["mol_dir"] = str(self._mol_dir)
        return row

    @staticmethod
    def _load_msa_entry(msas: list | None):
        """Load and parse a single MSA entry from a list of MSA descriptors."""
        if not msas:
            return None
        first = msas[0]
        if first is None:
            return None
        if first.get("sequences") or first.get("raw"):
            return first
        content = first.get("content") if isinstance(first, dict) else None
        if content is None and isinstance(first, dict) and first.get("path"):
            with open(first["path"]) as f:
                content = f.read()
        if content:
            from io import StringIO

            from tensorrt_bionemo.data.parsers.a3m import parse_a3m_content

            return parse_a3m_content(StringIO(content))
        return None
