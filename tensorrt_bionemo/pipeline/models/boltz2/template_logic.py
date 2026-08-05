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

"""Boltz-2 structural template logic — parse, chain selection, residue alignment.

Fresh TRT-BNM reimplementation of the OSS ``boltz`` v2.2.1 template pipeline
(direct path: a template mmCIF/PDB per protein chain). Covers sequence alignment
(``get_global_alignment_score`` / ``get_local_alignments``) and chain assignment
(``get_template_records_from_{search,matching}``, via ``linear_sum_assignment``
or explicit 1:1 zip); the offsets it returns feed the token-index mapping
``offset = template_st - query_st``. No OSS imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio import Align
from scipy.optimize import linear_sum_assignment

from tensorrt_bionemo.data.utils import normalize_gemmi_structure, read_gemmi_structure

from . import const
from .const import Atom, Chain, EnsembleDtype, Residue, Structure, Token, chain_type_ids
from .structure import _parse_modified_residue
from .tokenizer_logic import _IDENTITY_ROT, _ZERO_T, compute_frame


@dataclass(frozen=True)
class Alignment:
    """One ungapped local alignment block (half-open residue-index ranges).

    Mirrors OSS ``Alignment`` (query/template start-end from the alignment
    coordinate matrix). The featurizer reduces this to a single integer offset
    ``offset = template_st - query_st``.
    """

    query_st: int
    query_en: int
    template_st: int
    template_en: int


@dataclass(frozen=True)
class TemplateMatch:
    """A query-chain ↔ template-chain match with alignment offsets.

    Mirrors the alignment-carrying fields of upstream
    ``boltz.data.types.TemplateInfo``. ``name`` is the template id
    (file stem).
    """

    name: str
    query_chain: str
    query_st: int
    query_en: int
    template_chain: str
    template_st: int
    template_en: int
    force: bool = False
    threshold: float = float("inf")


def global_alignment_score(query: str, template: str) -> float:
    """Global blastp-scored alignment score between two sequences.

    Reimplements upstream ``get_global_alignment_score``
    (``boltz/data/parse/schema.py``): a
    Biopython ``PairwiseAligner(scoring="blastp")`` in global mode, returning the
    top alignment's score. Used to build the chain-assignment cost matrix.
    """
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    return float(aligner.align(query, template)[0].score)


def local_alignments(query: str, template: str) -> list[Alignment]:
    """Ungapped local alignment blocks between query and template sequences.

    Reimplements upstream ``get_local_alignments``
    (``boltz/data/parse/schema.py``): a blastp-scored
    ``PairwiseAligner`` in local mode with gap open/extend = ``-1000`` (so the
    alignment is effectively ungapped). Returns one :class:`Alignment` per block,
    taken from the alignment coordinate matrix rows (row 0 = query, row 1 =
    template).
    """
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "local"
    aligner.open_gap_score = -1000
    aligner.extend_gap_score = -1000

    out: list[Alignment] = []
    for result in aligner.align(query, template):
        coords = result.coordinates
        out.append(
            Alignment(
                query_st=int(coords[0][0]),
                query_en=int(coords[0][1]),
                template_st=int(coords[1][0]),
                template_en=int(coords[1][1]),
            )
        )
    return out


def template_records_from_search(
    template_id: str,
    chain_ids: list[str],
    sequences: dict[str, str],
    template_chain_ids: list[str],
    template_sequences: dict[str, str],
    force: bool = False,
    threshold: float | None = None,
) -> list[TemplateMatch]:
    """Auto-assign query chains to template chains, then align each pair.

    Reimplements upstream ``get_template_records_from_search``
    (``boltz/data/parse/schema.py``):
    build a ``len(chain_ids) x len(template_chain_ids)`` global-score matrix, solve
    the optimal assignment with ``linear_sum_assignment(..., maximize=True)``, then
    emit a :class:`TemplateMatch` per local-alignment block of each assigned pair.
    """
    score_matrix = [
        [global_alignment_score(sequences[cid], template_sequences[tcid]) for tcid in template_chain_ids]
        for cid in chain_ids
    ]

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    records: list[TemplateMatch] = []
    thr = float("inf") if threshold is None else threshold
    for r, c in zip(row_ind, col_ind, strict=True):
        cid = chain_ids[r]
        tcid = template_chain_ids[c]
        for aln in local_alignments(sequences[cid], template_sequences[tcid]):
            records.append(
                TemplateMatch(
                    name=template_id,
                    query_chain=cid,
                    query_st=aln.query_st,
                    query_en=aln.query_en,
                    template_chain=tcid,
                    template_st=aln.template_st,
                    template_en=aln.template_en,
                    force=force,
                    threshold=thr,
                )
            )
    return records


def template_records_from_matching(
    template_id: str,
    chain_ids: list[str],
    sequences: dict[str, str],
    template_chain_ids: list[str],
    template_sequences: dict[str, str],
    force: bool = False,
    threshold: float | None = None,
) -> list[TemplateMatch]:
    """Align an explicit 1:1 query-chain ↔ template-chain mapping.

    Reimplements upstream ``get_template_records_from_matching``
    (``boltz/data/parse/schema.py``):
    zip the two chain-id lists (equal length, user-specified) and emit a
    :class:`TemplateMatch` per local-alignment block, skipping the assignment step.
    """
    records: list[TemplateMatch] = []
    thr = float("inf") if threshold is None else threshold
    for cid, tcid in zip(chain_ids, template_chain_ids, strict=False):
        for aln in local_alignments(sequences[cid], template_sequences[tcid]):
            records.append(
                TemplateMatch(
                    name=template_id,
                    query_chain=cid,
                    query_st=aln.query_st,
                    query_en=aln.query_en,
                    template_chain=tcid,
                    template_st=aln.template_st,
                    template_en=aln.template_en,
                    force=force,
                    threshold=thr,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Template structure parse (gemmi) + tokenization (real coords).
# Reuses TRT's residue definitions (const.ref_atoms/token_ids/res_to_*_atom_id)
# so res_type/center/disto are produced exactly as they are for the query
# structure; the only template-specific part is overlaying the CIF's real
# coords + per-atom presence.
# ---------------------------------------------------------------------------
_POLYMER_TYPE_TO_CHAIN_TYPE = {
    "PeptideL": "PROTEIN",
    "Dna": "DNA",
    "Rna": "RNA",
}


_MODIFIED_MOL_CACHE: dict = {}


def _load_modified_mol(name: str, mol_dir):
    """Load a modified-residue CCD component mol, cached per ``(mol_dir, name)``.

    Thin cached wrapper over :func:`tensorrt_bionemo.data.utils.load_component_mol`
    (which validates ``name`` against path-traversal and confines the load to
    ``mol_dir``). Returns ``None`` when ``mol_dir`` is unset or the component is
    missing (caller then falls back to the UNK path).
    """
    if mol_dir is None:
        return None
    key = (str(mol_dir), name)
    if key in _MODIFIED_MOL_CACHE:
        return _MODIFIED_MOL_CACHE[key]
    from tensorrt_bionemo.data.utils import load_component_mol

    mol = load_component_mol(mol_dir, name)
    _MODIFIED_MOL_CACHE[key] = mol
    return mol


def _parse_template_polymer(polymer, polymer_type, sequence: list, chain_id: str, mol_dir=None) -> dict:
    """Parse one gemmi polymer into a residue list with real coords overlaid.

    Mirrors OSS ``parse_polymer``: align the full sequence to the polymer
    residues, then for each canonical atom (``const.ref_atoms``) overlay the
    CIF coordinate (present) or mark absent. Applies the OSS MSE->MET and
    ARG NH1/NH2 quirks (harmless for the emitted tensors but kept for fidelity).
    """
    import gemmi

    sequence = [gemmi.Entity.first_mon(item) for item in sequence]
    result = gemmi.align_sequence_to_polymer(sequence, polymer, polymer_type, gemmi.AlignmentScoring())

    ref_res = set(const.tokens)
    i = 0
    residues = []
    for j, match in enumerate(result.match_string):
        res_name = sequence[j]
        res = None
        name_to_atom: dict = {}
        if match == "|":
            res = polymer[i]
            name_to_atom = {a.name.upper(): a for a in res}
            i += 1

        # Map MSE to MET, put the selenium atom in the sulphur column.
        if res_name == "MSE":
            res_name = "MET"
            if "SE" in name_to_atom:
                name_to_atom["SD"] = name_to_atom["SE"]
        elif res_name not in ref_res:
            # Non-standard residue with a CCD mol: parse like OSS
            # parse_ccd_residue (heavy-atom order, atom_center/disto=0, UNK).
            # Fall back to a plain UNK atom set when no CCD mol is available.
            ref_mol = _load_modified_mol(res_name, mol_dir)
            if ref_mol is not None:
                residues.append(_parse_modified_residue(res_name, ref_mol, res, j))
                continue
            res_name = "UNK"

        atom_names = const.ref_atoms[res_name]
        atoms = []
        for atom_name in atom_names:
            atom = name_to_atom.get(atom_name)
            if atom is not None:
                coords = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                present = True
            else:
                coords = (0.0, 0.0, 0.0)
                present = False
            atoms.append((atom_name, coords, present))

        # Fix ARG NH1/NH2 mislabeling (OSS parse_polymer quirk).
        if (res is not None) and (res_name == "ARG"):
            names = list(atom_names)
            cd = atoms[names.index("CD")]
            nh1 = atoms[names.index("NH1")]
            nh2 = atoms[names.index("NH2")]
            if cd[2] and nh1[2] and nh2[2]:
                cd_c = np.array(cd[1])
                nh1_c = np.array(nh1[1])
                nh2_c = np.array(nh2[1])
                if np.linalg.norm(nh1_c - cd_c) > np.linalg.norm(nh2_c - cd_c):
                    atoms[names.index("NH1")] = (nh1[0], nh2[1], nh1[2])
                    atoms[names.index("NH2")] = (nh2[0], nh1[1], nh2[2])

        residues.append(
            {
                "name": res_name,
                "res_type": const.token_ids[res_name],
                "res_idx": j,
                "atoms": atoms,
                "atom_center": const.res_to_center_atom_id[res_name],
                "atom_disto": const.res_to_disto_atom_id[res_name],
                "is_present": res is not None,
                "is_standard": True,
            }
        )

    chain_type = chain_type_ids[_POLYMER_TYPE_TO_CHAIN_TYPE[polymer_type.name]]
    return {
        "name": chain_id,
        "type": chain_type,
        "residues": residues,
        "sequence": gemmi.one_letter_code(sequence),
    }


def parse_template_structure(
    source: str,
    fmt: str = "mmcif",
    from_content: bool = False,
    mol_dir=None,
) -> tuple[Structure, dict[str, str]]:
    """Parse a template mmCIF/PDB into a TRT :class:`Structure` (real coords).

    ``source`` is a filesystem path, or the raw file text when
    ``from_content=True`` (as carried by ``TemplateParsed["content"]``).
    Returns ``(structure, sequences)`` where ``sequences`` maps each protein
    chain name (gemmi subchain id, as OSS ``parse_mmcif`` names chains) to its
    one-letter sequence. Only polymer (protein/DNA/RNA) chains are emitted;
    atom coordinates are the CIF's real coordinates (``struct.coords`` and
    per-atom ``coords``), with ``is_present`` marking unresolved atoms.
    """
    st = read_gemmi_structure(source, fmt, from_content)
    normalize_gemmi_structure(st)

    # Map subchain id -> entity (mirror OSS parse_mmcif entity resolution).
    entities: dict[str, object] = {}
    for entity in st.entities:
        if entity.entity_type.name == "Water":
            continue
        for subchain_id in entity.subchains:
            entities[subchain_id] = entity

    parsed_chains: list[dict] = []
    for raw_chain in st[0].subchains():
        subchain_id = raw_chain.subchain_id()
        entity = entities.get(subchain_id)
        if entity is None or entity.entity_type.name != "Polymer":
            continue
        if entity.polymer_type.name not in _POLYMER_TYPE_TO_CHAIN_TYPE:
            continue
        parsed_chains.append(
            _parse_template_polymer(
                raw_chain, entity.polymer_type, list(entity.full_sequence), subchain_id, mol_dir=mol_dir
            )
        )

    if not parsed_chains:
        msg = "No polymer chains parsed from template!"
        raise ValueError(msg)

    # Flatten into TRT Structure tables.
    all_atoms: list[Atom] = []
    all_residues: list[Residue] = []
    all_chains: list[Chain] = []
    sequences: dict[str, str] = {}
    coords_list: list[tuple[float, float, float]] = []

    global_atom_idx = 0
    global_res_idx = 0
    entity_key_to_id: dict[str, int] = {}
    sym_count: dict[int, int] = {}

    for asym_id, pc in enumerate(parsed_chains):
        seq = pc["sequence"]
        entity_id = entity_key_to_id.setdefault(seq, len(entity_key_to_id))
        sym_id = sym_count.get(entity_id, 0)
        sym_count[entity_id] = sym_id + 1

        chain_atom_start = global_atom_idx
        chain_res_start = global_res_idx
        chain_atom_count = 0
        chain_res_count = 0

        for res in pc["residues"]:
            atom_center_global = global_atom_idx + res["atom_center"]
            atom_disto_global = global_atom_idx + res["atom_disto"]
            for atom_name, coords, present in res["atoms"]:
                all_atoms.append(
                    Atom(
                        name=atom_name,
                        element=0,
                        charge=0,
                        coords=coords,
                        conformer=coords,
                        is_present=present,
                        chirality=0,
                    )
                )
                coords_list.append(coords)
                global_atom_idx += 1
                chain_atom_count += 1
            all_residues.append(
                Residue(
                    name=res["name"],
                    res_type=res["res_type"],
                    res_idx=res["res_idx"],
                    atom_idx=global_atom_idx - len(res["atoms"]),
                    atom_num=len(res["atoms"]),
                    atom_center=atom_center_global,
                    atom_disto=atom_disto_global,
                    is_standard=res.get("is_standard", True),
                    is_present=res["is_present"],
                )
            )
            global_res_idx += 1
            chain_res_count += 1

        all_chains.append(
            Chain(
                name=pc["name"],
                mol_type=pc["type"],
                entity_id=entity_id,
                sym_id=sym_id,
                asym_id=asym_id,
                atom_idx=chain_atom_start,
                atom_num=chain_atom_count,
                res_idx=chain_res_start,
                res_num=chain_res_count,
                cyclic_period=0,
            )
        )
        if pc["type"] == chain_type_ids["PROTEIN"]:
            sequences[pc["name"]] = seq

    n_atoms = len(all_atoms)
    coords = np.asarray(coords_list, dtype=np.float32).reshape(n_atoms, 3)
    structure = Structure(
        atoms=all_atoms,
        bonds=[],
        residues=all_residues,
        chains=all_chains,
        coords=coords,
        ensemble=np.array([(0, n_atoms)], dtype=EnsembleDtype),
        mask=np.ones(len(all_chains), dtype=bool),
        bfactor=np.zeros(n_atoms, dtype=np.float32),
        plddt=np.ones(n_atoms, dtype=np.float32),
    )
    return structure, sequences


def tokenize_template(struct: Structure) -> list[Token]:
    """Tokenize a parsed template :class:`Structure` using REAL coordinates.

    Mirrors OSS ``tokenize_structure``, but the backbone frame and center/disto
    coords come from the template's real coordinates, not the ideal conformer.
    One token per residue.
    """
    tokens: list[Token] = []
    coords = struct.coords
    token_idx = 0

    chains = [c for c, m in zip(struct.chains, struct.mask, strict=True) if m]
    for chain in chains:
        is_protein = chain.mol_type == chain_type_ids["PROTEIN"]
        res_start = chain.res_idx
        res_end = chain.res_idx + chain.res_num
        for res in struct.residues[res_start:res_end]:
            center = struct.atoms[res.atom_center]
            disto = struct.atoms[res.atom_disto]
            is_present = bool(res.is_present and center.is_present)
            is_disto_present = bool(res.is_present and disto.is_present)
            center_coords = (
                float(coords[res.atom_center, 0]),
                float(coords[res.atom_center, 1]),
                float(coords[res.atom_center, 2]),
            )
            disto_coords = (
                float(coords[res.atom_disto, 0]),
                float(coords[res.atom_disto, 1]),
                float(coords[res.atom_disto, 2]),
            )

            frame_rot = _IDENTITY_ROT
            frame_t = _ZERO_T
            frame_mask = False
            # Upstream computes the backbone frame only for STANDARD
            # residues; a modified residue (is_standard=False, e.g. CSO) is
            # tokenized at residue level with frame_mask=False (see
            # ``tokenize_structure`` in ``boltz/data/tokenize/boltz2.py``).
            if is_protein and res.is_standard and res.atom_num >= 3:
                a0 = struct.atoms[res.atom_idx]
                a1 = struct.atoms[res.atom_idx + 1]
                a2 = struct.atoms[res.atom_idx + 2]
                frame_mask = bool(a0.is_present and a1.is_present and a2.is_present)
                if frame_mask:
                    frame_rot, frame_t = compute_frame(
                        tuple(coords[res.atom_idx]),
                        tuple(coords[res.atom_idx + 1]),
                        tuple(coords[res.atom_idx + 2]),
                    )

            tokens.append(
                Token(
                    token_idx=token_idx,
                    atom_idx=res.atom_idx,
                    atom_num=res.atom_num,
                    res_idx=res.res_idx,
                    res_type=res.res_type,
                    res_name=res.name,
                    sym_id=chain.sym_id,
                    asym_id=chain.asym_id,
                    entity_id=chain.entity_id,
                    mol_type=chain.mol_type,
                    center_idx=res.atom_center,
                    disto_idx=res.atom_disto,
                    center_coords=center_coords,
                    disto_coords=disto_coords,
                    resolved_mask=is_present,
                    disto_mask=is_disto_present,
                    modified=False,
                    frame_rot=frame_rot,
                    frame_t=frame_t,
                    frame_mask=frame_mask,
                    cyclic_period=chain.cyclic_period,
                    affinity_mask=False,
                )
            )
            token_idx += 1
    return tokens


# ---------------------------------------------------------------------------
# Template featurization (mirrors OSS compute/process_template_features)
# ---------------------------------------------------------------------------


def compute_template_features(
    query_tokens: list[Token],
    query_chain_asym_ids: list[int],
    tmpl_rows: list[dict],
    num_tokens: int,
) -> dict:
    """Per-token template feature arrays for one template row.

    Reimplements upstream ``compute_template_features``
    (``boltz/data/feature/featurizerv2.py``). ``tmpl_rows`` is a list of
    ``{"token": Token, "pdb_id": int, "q_idx": int}`` mapping a template token
    onto a query token index. Returns a dict of torch tensors (10 keys);
    ``query_to_template`` is allocated all-zeros exactly as OSS.
    """
    import torch
    from torch.nn.functional import one_hot

    res_type = np.zeros((num_tokens,), dtype=np.int64)
    frame_rot = np.zeros((num_tokens, 3, 3), dtype=np.float32)
    frame_t = np.zeros((num_tokens, 3), dtype=np.float32)
    cb_coords = np.zeros((num_tokens, 3), dtype=np.float32)
    ca_coords = np.zeros((num_tokens, 3), dtype=np.float32)
    frame_mask = np.zeros((num_tokens,), dtype=np.float32)
    cb_mask = np.zeros((num_tokens,), dtype=np.float32)
    template_mask = np.zeros((num_tokens,), dtype=np.float32)
    query_to_template = np.zeros((num_tokens,), dtype=np.int64)
    visibility_ids = np.zeros((num_tokens,), dtype=np.float32)

    asym_id_to_pdb_id: dict[int, int] = {}
    for row in tmpl_rows:
        idx = row["q_idx"]
        pdb_id = row["pdb_id"]
        token = row["token"]
        query_token = query_tokens[idx]
        asym_id_to_pdb_id[query_token.asym_id] = pdb_id
        res_type[idx] = token.res_type
        frame_rot[idx] = np.asarray(token.frame_rot, dtype=np.float32).reshape(3, 3)
        frame_t[idx] = np.asarray(token.frame_t, dtype=np.float32)
        cb_coords[idx] = np.asarray(token.disto_coords, dtype=np.float32)
        ca_coords[idx] = np.asarray(token.center_coords, dtype=np.float32)
        cb_mask[idx] = float(token.disto_mask)
        frame_mask[idx] = float(token.frame_mask)
        template_mask[idx] = 1.0

    q_asym = np.asarray([t.asym_id for t in query_tokens], dtype=np.int64)

    # Set visibility_id for templated chains.
    for asym_id, pdb_id in asym_id_to_pdb_id.items():
        visibility_ids[q_asym == asym_id] = pdb_id

    # Set visibility for non-templated chains (sentinel negative id).
    for asym_id in np.unique(np.asarray(query_chain_asym_ids, dtype=np.int64)):
        if asym_id not in asym_id_to_pdb_id:
            visibility_ids[q_asym == asym_id] = -1 - asym_id

    res_type_t = torch.from_numpy(res_type)
    res_type_t = one_hot(res_type_t, num_classes=const.num_tokens)

    return {
        "template_restype": res_type_t,
        "template_frame_rot": torch.from_numpy(frame_rot),
        "template_frame_t": torch.from_numpy(frame_t),
        "template_cb": torch.from_numpy(cb_coords),
        "template_ca": torch.from_numpy(ca_coords),
        "template_mask_cb": torch.from_numpy(cb_mask),
        "template_mask_frame": torch.from_numpy(frame_mask),
        "template_mask": torch.from_numpy(template_mask),
        "query_to_template": torch.from_numpy(query_to_template),
        "visibility_ids": torch.from_numpy(visibility_ids),
    }


def process_template_features(
    query_tokens: list[Token],
    query_chain_asym_ids: list[int],
    chain_name_to_asym_id: dict[str, int],
    template_matches: list[TemplateMatch],
    template_structures: dict[str, Structure],
    template_tokens: dict[str, list[Token]],
    max_tokens: int,
) -> dict:
    """Stack per-template features across all templates (mirrors OSS).

    Reimplements OSS ``process_template_features``: group matches by template
    name, compute ``offset = template_st - query_st`` per match, map template
    tokens onto query indices, build per-row features, attach the
    ``template_force`` / ``template_force_threshold`` scalars (from the LAST
    match, per the OSS variable-leak quirk), then stack across templates.
    """
    import torch

    name_to_templates: dict[str, list[TemplateMatch]] = {}
    for m in template_matches:
        name_to_templates.setdefault(m.name, []).append(m)

    template_features = []
    for template_id, (name, matches) in enumerate(name_to_templates.items()):
        row_tokens: list[dict] = []
        tmpl_struct = template_structures[name]
        tmpl_tokens = template_tokens[name]
        tmpl_chain_name_to_asym = {c.name: c.asym_id for c in tmpl_struct.chains}

        m: TemplateMatch | None = None
        for m in matches:
            offset = m.template_st - m.query_st

            cid = chain_name_to_asym_id[m.query_chain]
            q_toks = [t for t in query_tokens if t.asym_id == cid]
            q_indices = {t.res_idx: t.token_idx for t in q_toks}

            tcid = tmpl_chain_name_to_asym[m.template_chain]
            toks = [t for t in tmpl_tokens if t.asym_id == tcid]
            toks = [t for t in toks if (t.res_idx - offset) in q_indices]
            for t in toks:
                q_idx = q_indices[t.res_idx - offset]
                row_tokens.append({"token": t, "pdb_id": template_id, "q_idx": q_idx})

        row_features = compute_template_features(query_tokens, query_chain_asym_ids, row_tokens, max_tokens)
        row_features["template_force"] = torch.tensor(m.force)
        row_features["template_force_threshold"] = torch.tensor(
            m.threshold if m.threshold is not None else float("inf"),
            dtype=torch.float32,
        )
        template_features.append(row_features)

    out = {}
    for k in template_features[0]:
        out[k] = torch.stack([f[k] for f in template_features])
    return out


def build_template_features_from_row(
    query_tokens: list[Token],
    query_chain_asym_ids: list[int],
    query_name_to_asym: dict[str, int],
    templates_row: list[dict],
    num_tokens: int,
    mol_dir=None,
    max_templates: int | None = None,
) -> dict | None:
    """Build the stacked template features from a threaded ``row["templates"]``.

    ``templates_row`` (built by ``Boltz2ContextGenerator``) is a list, one entry
    per templated query polymer::

        {"chain_ids": [...], "sequence": query seq, "templates": [TemplateParsed]}

    Each ``TemplateParsed`` carries ``content`` (CIF/PDB text), ``format`` and an
    optional template ``chain_id``. Templates are grouped by content hash so the
    same file used across chains lands in ONE stacked T slot (first-seen order,
    per OSS ``process_template_features``). Returns ``None`` when no records are
    produced (caller uses the dummy path).
    """
    import hashlib

    # Parse each distinct template CONTENT once, keyed by a stable name.
    parsed_cache: dict[str, tuple[Structure, dict[str, str], list[Token]]] = {}
    matches: list[TemplateMatch] = []

    for spec in templates_row:
        query_seq = spec.get("sequence") or ""
        chain_ids = spec.get("chain_ids") or []
        for tmpl in spec.get("templates") or []:
            content = tmpl.get("content")
            if content is None:
                continue
            fmt = tmpl.get("format") or "cif"
            name = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
            if name not in parsed_cache:
                tmpl_struct, tmpl_seqs = parse_template_structure(content, fmt=fmt, from_content=True, mol_dir=mol_dir)
                parsed_cache[name] = (
                    tmpl_struct,
                    tmpl_seqs,
                    tokenize_template(tmpl_struct),
                )
            _, tmpl_seqs, _ = parsed_cache[name]
            tmpl_chain = tmpl.get("chain_id")

            if tmpl_chain is not None:
                # Explicit 1:1 mapping: each query chain uses the named
                # template chain. When the caller names a template chain it
                # must be honoured verbatim, so no Hungarian assignment or
                # auto-selection is applied on this branch.
                for query_chain in chain_ids:
                    matches.extend(
                        template_records_from_matching(
                            template_id=name,
                            chain_ids=[query_chain],
                            sequences={query_chain: query_seq},
                            template_chain_ids=[tmpl_chain],
                            template_sequences=tmpl_seqs,
                        )
                    )
            else:
                # Auto-select: ONE Hungarian assignment across all query chains
                # of this entity (OSS get_template_records_from_search) so
                # template chains map 1:1, not all grabbing the same best chain.
                matches.extend(
                    template_records_from_search(
                        template_id=name,
                        chain_ids=list(chain_ids),
                        sequences=dict.fromkeys(chain_ids, query_seq),
                        template_chain_ids=list(tmpl_seqs.keys()),
                        template_sequences=tmpl_seqs,
                    )
                )

    if not matches:
        return None

    # Optional memory cap (DEVIATION from OSS, which stacks all templates): keep
    # only the first ``max_templates`` distinct groups (first-seen; OSS has no
    # ranking) to bound the T dim and the module's T*N^2 pair memory. ``None``
    # (default) = no cap = OSS-faithful.
    if max_templates is not None and max_templates > 0:
        seen_names: list[str] = []
        for m in matches:
            if m.name not in seen_names:
                seen_names.append(m.name)
        keep = set(seen_names[:max_templates])
        matches = [m for m in matches if m.name in keep]
        parsed_cache = {n: c for n, c in parsed_cache.items() if n in keep}

    template_structures = {name: c[0] for name, c in parsed_cache.items()}
    template_tokens = {name: c[2] for name, c in parsed_cache.items()}
    return process_template_features(
        query_tokens=query_tokens,
        query_chain_asym_ids=query_chain_asym_ids,
        chain_name_to_asym_id=query_name_to_asym,
        template_matches=matches,
        template_structures=template_structures,
        template_tokens=template_tokens,
        max_tokens=num_tokens,
    )
