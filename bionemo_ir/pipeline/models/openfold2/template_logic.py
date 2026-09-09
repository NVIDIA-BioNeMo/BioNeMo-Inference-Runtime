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

"""Direct-CIF template feature construction for OpenFold2.

The user supplies CIF content directly, so this module only performs the
sequence alignment, chain selection, and atom37 packing that happen after a
template search. Chain IDs at the API boundary are PDB author-chain IDs. CIF
label asym IDs and label sequence positions remain internal parsing details.

Ported from OpenFold ``mmcif_parsing.py`` / ``templates.py`` pinned at
be2ec1841f16c966c65ae0e7599ebbadc725757d; the public functions cite exact
line ranges, and BioIR-specific policy is called out where it diverges.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bionemo_ir.data.tools.kalign import run_kalign
from bionemo_ir.data.tools.template_alignment import calculate_ids_hit, seq_identity_and_coverage
from bionemo_ir.pipeline.models.openfold2 import const as rc

logger = logging.getLogger(__name__)

_WEAK_TEMPLATE_SCORE = 0.1
# Warn (visibility only, no truncation) when a request supplies far more
# templates than it can use: every one is aligned before all but the top
# ``max_templates`` are kept, so a large caller-driven count is wasted work.
_INPUT_TEMPLATE_WARN_FACTOR = 4


@dataclass(frozen=True)
class ChainTemplateData:
    """Canonical sequence and centered atom37 coordinates for one chain."""

    block_name: str
    chain_id: str
    canonical_seq: str
    res_name_by_pos: dict[int, str]
    coords_by_pos: dict[int, dict[str, np.ndarray]]


@dataclass(frozen=True)
class SelectedTemplate:
    """One CIF chain selected and aligned to the query sequence."""

    chain_id: str
    idx_map: np.ndarray
    sequence_identity: float
    query_coverage: float
    score: float
    chain_data: ChainTemplateData


def stable_top_k[T](items: Sequence[T], k: int, key: Callable[[T], float]) -> list[T]:
    """Return the highest-scoring ``k`` items with stable input-order ties."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    ranked = sorted(enumerate(items), key=lambda indexed: (-float(key(indexed[1])), indexed[0]))
    return [item for _, item in ranked[:k]]


def align_query_to_template_chain(
    query_seq: str,
    chain_data: ChainTemplateData,
) -> tuple[np.ndarray, float, float]:
    """Align a query to a CIF chain and return its 1-based residue mapping."""
    if not query_seq:
        raise ValueError("query_seq must not be empty")
    if not chain_data.canonical_seq:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    aligned = run_kalign([query_seq, chain_data.canonical_seq])
    if len(aligned) != 2:
        raise ValueError(f"Expected two aligned sequences, got {len(aligned)}")
    if len(aligned[0]) != len(aligned[1]):
        raise ValueError("Kalign returned sequences with unequal lengths")

    query_aligned = np.asarray(list(aligned[0]), dtype="<U1")
    template_aligned = np.asarray(list(aligned[1]), dtype="<U1")
    sequence_identity, query_coverage = seq_identity_and_coverage(query_aligned, template_aligned, query_seq)
    query_ids, template_ids = calculate_ids_hit(query_aligned, template_aligned)
    both_present = (query_ids != -1) & (template_ids != -1)
    idx_map = np.stack([query_ids[both_present], template_ids[both_present]], axis=-1)
    return idx_map.astype(np.int64), sequence_identity, query_coverage


def _clean_canonical_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def _category_column(category, name: str, dtype: type) -> np.ndarray | None:
    if name not in category:
        return None
    return category[name].as_array(dtype)


def _entity_polymer_data(block) -> tuple[dict[str, str], dict[str, str]]:
    """Return canonical sequences and polymer types keyed by entity ID."""
    if "entity_poly" not in block:
        return {}, {}
    category = block["entity_poly"]
    entity_ids = _category_column(category, "entity_id", str)
    sequences = _category_column(category, "pdbx_seq_one_letter_code_can", str)
    polymer_types = _category_column(category, "type", str)
    if entity_ids is None:
        return {}, {}

    sequence_by_entity: dict[str, str] = {}
    type_by_entity: dict[str, str] = {}
    for index, entity_id in enumerate(entity_ids.tolist()):
        entity_id = str(entity_id)
        if sequences is not None:
            sequence_by_entity[entity_id] = _clean_canonical_sequence(str(sequences[index]))
        if polymer_types is not None:
            type_by_entity[entity_id] = str(polymer_types[index]).lower()
    return sequence_by_entity, type_by_entity


def _modified_residue_parents(block) -> dict[str, str]:
    """Return modified 3-letter residue names mapped to standard parents."""
    parents = {"MSE": "MET"}
    if "chem_comp" not in block:
        return parents
    category = block["chem_comp"]
    residue_names = _category_column(category, "id", str)
    parent_names = _category_column(category, "mon_nstd_parent_comp_id", str)
    if residue_names is None or parent_names is None:
        return parents

    for residue_name, parent_name in zip(residue_names.tolist(), parent_names.tolist(), strict=True):
        parent_name = str(parent_name).strip().upper()
        if parent_name not in {"", ".", "?"}:
            parents[str(residue_name).strip().upper()] = parent_name.split(",", maxsplit=1)[0]
    return parents


def _residue_to_one_letter(residue_name: str, modified_parents: Mapping[str, str]) -> str:
    residue_name = residue_name.strip().upper()
    parent = modified_parents.get(residue_name, residue_name)
    return rc.restype_3to1.get(parent, "X")


def _author_chain_by_label(block) -> dict[str, str]:
    """Map label asym IDs to the PDB author-chain namespace."""
    mapping: dict[str, str] = {}
    if "pdbx_poly_seq_scheme" in block:
        scheme = block["pdbx_poly_seq_scheme"]
        labels = _category_column(scheme, "asym_id", str)
        authors = _category_column(scheme, "pdb_strand_id", str)
        if labels is not None and authors is not None:
            for label, author in zip(labels.tolist(), authors.tolist(), strict=True):
                author = str(author).strip()
                if author not in {"", ".", "?"}:
                    mapping.setdefault(str(label), author)

    if "atom_site" in block:
        atom_site = block["atom_site"]
        labels = _category_column(atom_site, "label_asym_id", str)
        authors = _category_column(atom_site, "auth_asym_id", str)
        if labels is not None and authors is not None:
            for label, author in zip(labels.tolist(), authors.tolist(), strict=True):
                author = str(author).strip()
                if author not in {"", ".", "?"}:
                    mapping.setdefault(str(label), author)
    return mapping


def _canonical_sequence_by_label(
    block,
) -> tuple[dict[str, str], dict[str, dict[int, str]], list[str]]:
    """Read protein sequence metadata in CIF label-asym order."""
    if "pdbx_poly_seq_scheme" not in block:
        return {}, {}, []

    scheme = block["pdbx_poly_seq_scheme"]
    labels = _category_column(scheme, "asym_id", str)
    entity_ids = _category_column(scheme, "entity_id", str)
    positions = _category_column(scheme, "seq_id", int)
    residue_names = _category_column(scheme, "mon_id", str)
    if labels is None or positions is None or residue_names is None:
        return {}, {}, []

    sequence_by_entity, type_by_entity = _entity_polymer_data(block)
    modified_parents = _modified_residue_parents(block)
    names_by_label: dict[str, dict[int, str]] = {}
    entity_by_label: dict[str, str] = {}
    label_order: list[str] = []
    for index, (label, position, residue_name) in enumerate(
        zip(labels.tolist(), positions.tolist(), residue_names.tolist(), strict=True)
    ):
        label = str(label)
        if label not in names_by_label:
            names_by_label[label] = {}
            label_order.append(label)
        names_by_label[label].setdefault(int(position), str(residue_name).upper())
        if entity_ids is not None:
            entity_by_label.setdefault(label, str(entity_ids[index]))

    canonical_by_label: dict[str, str] = {}
    protein_labels: list[str] = []
    for label in label_order:
        entity_id = entity_by_label.get(label)
        polymer_type = type_by_entity.get(entity_id, "")
        names = names_by_label[label]
        inferred_is_protein = any(_residue_to_one_letter(name, modified_parents) != "X" for name in names.values())
        if polymer_type and "polypeptide" not in polymer_type:
            continue
        if not polymer_type and not inferred_is_protein:
            continue

        max_position = max(names, default=0)
        entity_sequence = sequence_by_entity.get(entity_id, "")
        if len(entity_sequence) == max_position and all(residue.isalpha() for residue in entity_sequence):
            canonical = entity_sequence
        else:
            canonical = "".join(
                _residue_to_one_letter(names.get(position, "UNK"), modified_parents)
                for position in range(1, max_position + 1)
            )
        canonical_by_label[label] = canonical
        protein_labels.append(label)
    return canonical_by_label, names_by_label, protein_labels


# Exact parity:
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/mmcif_parsing.py#L497-L500
def _center_atom_coordinates(
    coords_by_pos: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    # Match OpenFold's dense ``[residue, atom37]`` reduction order. Floating
    # point centering is order-sensitive, and tiny coordinate differences can
    # become large values for geometrically degenerate masked torsions.
    coordinates = [
        coords_by_pos[position][atom_name]
        for position in sorted(coords_by_pos)
        for atom_name in rc.atom_types
        if atom_name in coords_by_pos[position]
    ]
    if not coordinates:
        return coords_by_pos
    center = np.stack(coordinates, axis=0).astype(np.float32).mean(axis=0)
    return {
        position: {
            atom_name: (coordinate.astype(np.float32) - center).astype(np.float32)
            for atom_name, coordinate in atoms.items()
        }
        for position, atoms in coords_by_pos.items()
    }


def _correct_arg_atom_names(residue_name: str, atoms: dict[str, np.ndarray]) -> None:
    if residue_name != "ARG" or not {"CD", "NH1", "NH2"}.issubset(atoms):
        return
    nh1_distance = np.linalg.norm(atoms["NH1"] - atoms["CD"])
    nh2_distance = np.linalg.norm(atoms["NH2"] - atoms["CD"])
    if nh1_distance > nh2_distance:
        atoms["NH1"], atoms["NH2"] = atoms["NH2"], atoms["NH1"]


# Adapted parity; this port uses Biotite and materializes per-chain records:
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/mmcif_parsing.py#L178-L305
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/mmcif_parsing.py#L435-L502
def extract_template_chains(content: str, fmt: str = "cif") -> dict[str, ChainTemplateData]:
    """Parse protein chains from CIF content, keyed by author-chain ID."""
    if fmt.lower() != "cif":
        raise ValueError(f"Only CIF templates are supported, got {fmt!r}")
    if not content or not content.strip():
        raise ValueError("Template CIF content must not be empty")

    from biotite.structure.io.pdbx import CIFFile, get_structure

    try:
        cif = CIFFile.read(io.StringIO(content))
    except Exception as error:
        raise ValueError("Could not parse template CIF content") from error
    if len(cif) != 1:
        raise ValueError(f"Expected one CIF data block, found {len(cif)}")
    block_name, block = next(iter(cif.items()))

    canonical_by_label, names_by_label, label_order = _canonical_sequence_by_label(block)
    author_by_label = _author_chain_by_label(block)

    atom_rows_by_label: dict[str, list[tuple[int, str, str, np.ndarray]]] = {}
    if "atom_site" in block:
        try:
            atom_array = get_structure(cif, model=1, use_author_fields=False, altloc="occupancy")
        except Exception as error:
            raise ValueError("Could not read atom_site from template CIF") from error
        for index in range(atom_array.array_length()):
            label = str(atom_array.chain_id[index])
            try:
                position = int(atom_array.res_id[index])
            except (TypeError, ValueError):
                continue
            if position < 1:
                continue
            atom_name = str(atom_array.atom_name[index]).strip().upper()
            residue_name = str(atom_array.res_name[index]).strip().upper()
            if residue_name == "MSE" and atom_name == "SE":
                atom_name = "SD"
            if atom_name not in rc.atom_order:
                continue
            coordinate = np.asarray(atom_array.coord[index], dtype=np.float32)
            if not np.all(np.isfinite(coordinate)):
                raise ValueError(
                    "Template CIF contains a non-finite coordinate for "
                    f"label chain {label!r}, residue {position}, atom "
                    f"{atom_name!r}"
                )
            atom_rows_by_label.setdefault(label, []).append((position, residue_name, atom_name, coordinate))
            if label not in label_order and label not in names_by_label:
                label_order.append(label)

    modified_parents = _modified_residue_parents(block)
    chains: dict[str, ChainTemplateData] = {}
    for label in label_order:
        rows = atom_rows_by_label.get(label, [])
        names = dict(names_by_label.get(label, {}))
        for position, residue_name, _, _ in rows:
            names.setdefault(position, residue_name)

        canonical_seq = canonical_by_label.get(label)
        if canonical_seq is None:
            max_position = max(names, default=0)
            canonical_seq = "".join(
                _residue_to_one_letter(names.get(position, "UNK"), modified_parents)
                for position in range(1, max_position + 1)
            )
            if not any(residue != "X" for residue in canonical_seq):
                continue
        if not canonical_seq:
            continue

        coords_by_pos: dict[int, dict[str, np.ndarray]] = {}
        for position, _, atom_name, coordinate in rows:
            coords_by_pos.setdefault(position, {})[atom_name] = coordinate
        for position, atoms in coords_by_pos.items():
            _correct_arg_atom_names(names.get(position, "UNK"), atoms)
        coords_by_pos = _center_atom_coordinates(coords_by_pos)

        author_chain_id = author_by_label.get(label, label)
        if author_chain_id in chains:
            raise ValueError(f"Author chain ID {author_chain_id!r} maps to multiple CIF chains")
        chains[author_chain_id] = ChainTemplateData(
            block_name=str(block_name),
            chain_id=author_chain_id,
            canonical_seq=canonical_seq,
            res_name_by_pos=names,
            coords_by_pos=coords_by_pos,
        )
    return chains


# BioIR policy for identity-times-coverage auto-selection; OSS mechanics:
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/templates.py#L292-L350
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/templates.py#L366-L502
def select_template_for_cif(
    query_seq: str,
    content: str,
    fmt: str = "cif",
    specified_chain_id: str | None = None,
    weak_match_threshold: float = _WEAK_TEMPLATE_SCORE,
) -> SelectedTemplate:
    """Select and align one CIF chain for a query sequence.

    Explicit author-chain requests are never rejected by score. Automatic
    selection requires a nonempty mutual alignment and warns for a weak score.
    """
    chains = extract_template_chains(content, fmt)
    if not chains:
        raise ValueError("Template CIF contains no protein chains")

    if specified_chain_id is not None:
        if specified_chain_id not in chains:
            raise ValueError(
                f"Template author chain {specified_chain_id!r} was not found; available chains: {sorted(chains)}"
            )
        candidates = [(specified_chain_id, chains[specified_chain_id])]
    else:
        candidates = list(chains.items())

    selected: SelectedTemplate | None = None
    for chain_id, chain_data in candidates:
        idx_map, sequence_identity, query_coverage = align_query_to_template_chain(query_seq, chain_data)
        if idx_map.shape[0] == 0:
            continue
        score = sequence_identity * query_coverage
        candidate = SelectedTemplate(
            chain_id=chain_id,
            idx_map=idx_map,
            sequence_identity=sequence_identity,
            query_coverage=query_coverage,
            score=score,
            chain_data=chain_data,
        )
        if selected is None or candidate.score > selected.score:
            selected = candidate

    if selected is None:
        chain_description = (
            f"author chain {specified_chain_id!r}" if specified_chain_id is not None else "every protein chain"
        )
        raise ValueError(f"Query has no mutually aligned residues with {chain_description}")

    if specified_chain_id is None and selected.score < weak_match_threshold:
        logger.warning(
            "Automatically selected weak template match %s_%s (identity=%.3f, coverage=%.3f, score=%.3f)",
            selected.chain_data.block_name,
            selected.chain_id,
            selected.sequence_identity,
            selected.query_coverage,
            selected.score,
        )
    return selected


def _template_value(template: Any, name: str, default: Any = None) -> Any:
    if isinstance(template, Mapping):
        return template.get(name, default)
    return getattr(template, name, default)


def _template_content(template: Any) -> str:
    content = _template_value(template, "content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Each supplied template must contain materialized CIF content")
    return content


def _empty_template_feats(num_residues: int) -> dict[str, np.ndarray]:
    """Match the existing OpenFold2 no-template feature shapes."""
    return {
        "template_aatype": np.zeros((0, num_residues, len(rc.restypes_with_x_and_gap)), dtype=np.float32),
        "template_all_atom_positions": np.zeros((0, num_residues, rc.atom_type_num, 3), dtype=np.float32),
        "template_all_atom_mask": np.zeros((0, num_residues, rc.atom_type_num), dtype=np.float32),
        "template_sum_probs": np.zeros((0, 1), dtype=np.float32),
        "template_domain_names": np.asarray([b""], dtype=object),
        "template_sequence": np.asarray([b""], dtype=object),
    }


def _aligned_template_sequence(query_length: int, selected: SelectedTemplate) -> str:
    aligned = ["-"] * query_length
    canonical = selected.chain_data.canonical_seq
    for query_position, template_position in selected.idx_map:
        query_index = int(query_position) - 1
        template_index = int(template_position) - 1
        if not (0 <= query_index < query_length and 0 <= template_index < len(canonical)):
            continue
        residue = canonical[template_index].upper()
        aligned[query_index] = residue if residue in rc.HHBLITS_AA_TO_ID else "X"
    return "".join(aligned)


# Adapted parity:
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/templates.py#L505-L705
def _pack_selected_template(
    query_sequence: str,
    selected: SelectedTemplate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes, bytes]:
    previous_ca = None
    previous_has_ca = False
    for position in range(1, len(selected.chain_data.canonical_seq) + 1):
        ca = selected.chain_data.coords_by_pos.get(position, {}).get("CA")
        has_ca = ca is not None
        if has_ca and previous_has_ca:
            distance = np.linalg.norm(ca - previous_ca)
            if distance > 150.0:
                raise ValueError(
                    f"Selected template {selected.chain_data.block_name}_"
                    f"{selected.chain_id} has adjacent C-alpha distance "
                    f"{distance:.3f} > 150.000 between residues "
                    f"{position - 1} and {position}"
                )
        if has_ca:
            previous_ca = ca
        previous_has_ca = has_ca

    num_residues = len(query_sequence)
    positions = np.zeros((num_residues, rc.atom_type_num, 3), dtype=np.float32)
    mask = np.zeros((num_residues, rc.atom_type_num), dtype=np.float32)

    for query_position, template_position in selected.idx_map:
        query_index = int(query_position) - 1
        if not 0 <= query_index < num_residues:
            continue
        atoms = selected.chain_data.coords_by_pos.get(int(template_position), {})
        for atom_name, coordinate in atoms.items():
            atom_index = rc.atom_order[atom_name]
            positions[query_index, atom_index] = coordinate
            mask[query_index, atom_index] = 1.0

    if np.sum(mask) < 5:
        raise ValueError(
            f"Selected template {selected.chain_data.block_name}_"
            f"{selected.chain_id} has fewer than 5 aligned atom37 "
            "coordinates"
        )

    aligned_sequence = _aligned_template_sequence(num_residues, selected)
    # OpenFold's populated direct-CIF features use int64 one-hot values. Keep
    # that boundary dtype so the features compare byte-exactly against
    # upstream OpenFold's output. The empty-template path
    # (`_empty_template_feats`) emits float32 instead, also matching
    # upstream.
    aatype = np.zeros((num_residues, 22), dtype=np.int64)
    for residue_index, residue in enumerate(aligned_sequence):
        aatype[residue_index, rc.HHBLITS_AA_TO_ID.get(residue, 20)] = 1.0

    domain_name = (f"{selected.chain_data.block_name.lower()}_{selected.chain_id}").encode()
    return (aatype, positions, mask, domain_name, aligned_sequence.encode("utf-8"))


# Adapted parity; direct-CIF orchestration with deterministic BioIR top-k:
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/templates.py#L950-L1014
# https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/templates.py#L1142-L1205
def build_template_feats(
    query_sequence: str,
    templates: Sequence[Any] | None,
    chain_id: str | None = None,
    *,
    max_templates: int = 4,
) -> dict[str, np.ndarray]:
    """Build OpenFold2's six base template features from direct CIF inputs.

    ``chain_id`` identifies the query chain for diagnostics only. Each template's
    own optional ``chain_id`` selects a PDB author chain in that CIF.
    """
    if not query_sequence:
        suffix = f" for query chain {chain_id!r}" if chain_id else ""
        raise ValueError(f"Query sequence must not be empty{suffix}")
    if max_templates < 0:
        raise ValueError(f"max_templates must be non-negative, got {max_templates}")
    if not templates or max_templates == 0:
        return _empty_template_feats(len(query_sequence))

    if len(templates) > max_templates * _INPUT_TEMPLATE_WARN_FACTOR:
        suffix = f" for query chain {chain_id!r}" if chain_id else ""
        logger.warning(
            "Received %d templates%s but max_templates=%d; every supplied "
            "template is aligned before all but the top %d are kept",
            len(templates),
            suffix,
            max_templates,
            max_templates,
        )

    selected_templates: list[SelectedTemplate] = []
    for template_index, template in enumerate(templates):
        try:
            content = _template_content(template)
            fmt = _template_value(template, "format", "cif") or "cif"
            specified_chain_id = _template_value(template, "chain_id")
            if specified_chain_id is not None and not isinstance(specified_chain_id, str):
                raise ValueError("chain_id must be a string or None")
            selected = select_template_for_cif(
                query_sequence,
                content,
                fmt=str(fmt),
                specified_chain_id=specified_chain_id,
            )
        except ValueError as error:
            suffix = f" for query chain {chain_id!r}" if chain_id else ""
            raise ValueError(f"Invalid supplied template at index {template_index}{suffix}: {error}") from error
        selected_templates.append(selected)

    selected_templates = stable_top_k(selected_templates, max_templates, key=lambda item: item.score)
    packed = [_pack_selected_template(query_sequence, selected) for selected in selected_templates]
    return {
        "template_aatype": np.stack([item[0] for item in packed]).astype(np.int64),
        "template_all_atom_positions": np.stack([item[1] for item in packed]).astype(np.float32),
        "template_all_atom_mask": np.stack([item[2] for item in packed]).astype(np.float32),
        "template_sum_probs": np.ones((len(packed), 1), dtype=np.float32),
        "template_domain_names": np.asarray([item[3] for item in packed], dtype=object),
        "template_sequence": np.asarray([item[4] for item in packed], dtype=object),
    }
