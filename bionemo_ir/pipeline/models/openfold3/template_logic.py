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
"""OpenFold3 direct-CIF template logic (protein-only).

Turns a user-supplied template CIF into per-query-token pseudo-beta /
backbone-frame coordinates, mirroring the OSS OpenFold-3 CIF-direct flow
(``CifDirectParser`` + ``map_token_pos_to_template_residues``).

Protein templates only: one token per residue, so an aligned residue's query
token is found by its 1-based query residue index. Alignment uses kalign (pip
``kalign-python``, the same aligner as OSS ``run_kalign``).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np

from bionemo_ir.data.tools.kalign import run_kalign
from bionemo_ir.data.tools.template_alignment import calculate_ids_hit, seq_identity_and_coverage

from .const import _PROTEIN_1TO3, TEMPLATE_CIF_DIRECT_MIN_SCORE

logger = logging.getLogger(__name__)

# 3-letter -> 1-letter protein map; non-standard/modified residues -> "X".
_PROTEIN_3TO1: dict[str, str] = {v: k for k, v in _PROTEIN_1TO3.items()}

# Backbone + pseudo-beta atom names read from the template structure.
_FRAME_ATOMS = ("N", "CA", "C")

# 20 amino acids + UNK. An aligned residue outside this set triggers the OSS
# "non-standard" cleaning branch (see ``resolve_template_idx_map``).
_STANDARD_PROTEIN_3: frozenset = frozenset(_PROTEIN_3TO1) | {"UNK"}


@dataclass
class ChainTemplateData:
    """Per-chain data extracted from a template CIF.

    Attributes:
        canonical_seq: 1-letter canonical sequence (1-based positions),
            including residues that are unresolved in the coordinates.
        res_name_by_pos: 1-based canonical position -> 3-letter residue name.
        coords_by_pos: 1-based canonical position -> {atom_name: (3,) np.ndarray}.
            Only resolved atoms are present; missing atoms -> NaN downstream.
    """

    canonical_seq: str
    res_name_by_pos: dict[int, str]
    coords_by_pos: dict[int, dict[str, np.ndarray]]


@dataclass
class SelectedTemplate:
    """A selected template chain aligned to a query chain."""

    chain_id: str
    idx_map: np.ndarray  # (n, 2): [query_res_idx(1-based), template_pos(1-based)]
    score: float
    chain_data: ChainTemplateData


# ---------------------------------------------------------------------------
# Alignment (kalign — same package as OSS run_kalign)
# ---------------------------------------------------------------------------


def align_query_to_template_chain(
    query_seq: str,
    chain_data: ChainTemplateData,
) -> tuple[np.ndarray, float, float]:
    """Align a query sequence to one template chain via kalign.

    Returns (idx_map, seq_id, q_cov) where idx_map is (n, 2) of
    [query_res_idx(1-based), template_canonical_pos(1-based)] for aligned,
    mutually-ungapped positions.
    """
    template_seq = chain_data.canonical_seq
    if not template_seq:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    aln = run_kalign([query_seq, template_seq])
    if len(aln) < 2 or not aln[0] or not aln[1]:
        return np.empty((0, 2), dtype=np.int64), 0.0, 0.0

    q_arr = np.fromiter(aln[0], dtype="<U1", count=len(aln[0]))
    t_arr = np.fromiter(aln[1], dtype="<U1", count=len(aln[1]))
    seq_id, q_cov = seq_identity_and_coverage(q_arr, t_arr, query_seq)
    q_hit, t_hit = calculate_ids_hit(q_arr, t_arr)

    idx_map = np.concatenate([q_hit[:, None], t_hit[:, None]], axis=1)
    # Keep only positions aligned on BOTH sides (drop -1 gaps).
    idx_map = idx_map[(idx_map[:, 0] != -1) & (idx_map[:, 1] != -1)]
    return idx_map.astype(np.int64), seq_id, q_cov


# ---------------------------------------------------------------------------
# CIF chain extraction (biotite)
# ---------------------------------------------------------------------------


def _canonical_seq_from_poly_scheme(block) -> dict[str, tuple[str, dict[int, str]]] | None:
    """Per-asym canonical sequence + residue names for a template CIF.

    Returns dict[label_asym_id -> (canonical_seq, {seq_id: res_name_3})] or
    ``None`` if ``pdbx_poly_seq_scheme`` is absent.

    ``canonical_seq`` comes from ``entity_poly.pdbx_seq_one_letter_code_can``
    (the sequence OSS aligns against), which uses the parent one-letter code for
    modified residues (e.g. MSE -> 'M'); a 3->1 mapping would emit 'X' and shift
    the alignment. Per-position 3-letter names come from ``mon_id`` keyed by
    ``seq_id``. Falls back to the 3->1 mapping when ``entity_poly`` is absent.
    """
    if "pdbx_poly_seq_scheme" not in block:
        return None
    scheme = block["pdbx_poly_seq_scheme"]
    asym = scheme["asym_id"].as_array(str)
    mon = scheme["mon_id"].as_array(str)
    seq_id = scheme["seq_id"].as_array(int)

    entity_seqs = _entity_canonical_seqs(block)

    out: dict[str, tuple[list, dict[int, str]]] = {}
    for a, m, s in zip(asym, mon, seq_id, strict=False):
        chars, names = out.setdefault(a, ([], {}))
        chars.append((int(s), _PROTEIN_3TO1.get(m, "X")))
        names[int(s)] = m
    result: dict[str, tuple[str, dict[int, str]]] = {}
    for a, (chars, names) in out.items():
        chars.sort(key=lambda x: x[0])
        seq = entity_seqs.get(a) or "".join(c for _, c in chars)
        result[a] = (seq, names)
    return result


def _entity_canonical_seqs(block) -> dict[str, str]:
    """asym_id -> canonical one-letter sequence from ``entity_poly``.

    Reads ``pdbx_seq_one_letter_code_can`` per entity (stripping newlines only,
    as OSS does) and maps each asym_id to its entity's sequence. Returns ``{}``
    if the required categories/columns are absent.
    """
    if "entity_poly" not in block or "pdbx_poly_seq_scheme" not in block:
        return {}
    ep = block["entity_poly"]
    if "pdbx_seq_one_letter_code_can" not in ep or "entity_id" not in ep:
        return {}
    ent_ids = ep["entity_id"].as_array(int)
    seqs = [s.replace("\n", "") for s in ep["pdbx_seq_one_letter_code_can"].as_array(str)]
    ent2seq = {int(e): s for e, s in zip(ent_ids.tolist(), seqs, strict=False)}

    scheme = block["pdbx_poly_seq_scheme"]
    if "entity_id" not in scheme:
        return {}
    asym = scheme["asym_id"].as_array(str)
    ent = scheme["entity_id"].as_array(int)
    out: dict[str, str] = {}
    for a, e in zip(asym.tolist(), ent.tolist(), strict=False):
        if a not in out and int(e) in ent2seq:
            out[a] = ent2seq[int(e)]
    return out


def extract_template_chains(content: str, fmt: str = "cif") -> dict[str, ChainTemplateData]:
    """Parse a template CIF into per-chain sequence + backbone/pseudo-beta coords.

    Uses biotite with ``use_author_fields=False`` so that ``chain_id`` is the
    label_asym_id and ``res_id`` is the 1-based label_seq_id (canonical
    position) — matching the numbering OSS template idx maps use.
    """
    from biotite.structure.io.pdbx import CIFFile, get_structure

    if fmt != "cif":
        raise ValueError(f"Only 'cif' template format is supported, got {fmt!r}")

    cif = CIFFile.read(io.StringIO(content))
    block = cif.block
    poly = _canonical_seq_from_poly_scheme(block)

    # altloc="occupancy" picks the highest-occupancy conformer, matching OSS.
    # Biotite's default ("first") picks altloc 'A' regardless and diverges.
    arr = get_structure(cif, model=1, use_author_fields=False, altloc="occupancy")
    # Heavy atoms of amino acids only.
    arr = arr[arr.element != "H"]

    chains: dict[str, ChainTemplateData] = {}
    # Include chains that appear only in the sequence metadata (zero atom_site
    # rows) so a valid specified_chain_id isn't treated as missing.
    chain_ids = set(map(str, np.unique(arr.chain_id)))
    if poly:
        chain_ids |= set(poly)

    for chain_id in sorted(chain_ids):
        chain_atoms = arr[arr.chain_id == chain_id]

        coords_by_pos: dict[int, dict[str, np.ndarray]] = {}
        resname_by_pos: dict[int, str] = {}
        for i in range(chain_atoms.array_length()):
            try:
                pos = int(chain_atoms.res_id[i])
            except (ValueError, TypeError):
                continue
            if pos < 1:
                continue
            aname = str(chain_atoms.atom_name[i])
            if aname not in _FRAME_ATOMS and aname != "CB":
                # Only backbone frame + pseudo-beta atoms are needed.
                resname_by_pos.setdefault(pos, str(chain_atoms.res_name[i]))
                continue
            coords_by_pos.setdefault(pos, {})[aname] = np.asarray(chain_atoms.coord[i], dtype=np.float64)
            resname_by_pos.setdefault(pos, str(chain_atoms.res_name[i]))

        if poly is not None and chain_id in poly:
            canonical_seq, names = poly[chain_id]
            # Prefer scheme res names (covers unresolved positions).
            for p, n in names.items():
                resname_by_pos.setdefault(p, n)
        else:
            # No pdbx_poly_seq_scheme: fall back to resolved-residue order
            # (deviation: unresolved residues absent, can shift numbering).
            positions = sorted(resname_by_pos)
            canonical_seq = "".join(_PROTEIN_3TO1.get(resname_by_pos[p], "X") for p in positions)

        chains[str(chain_id)] = ChainTemplateData(
            canonical_seq=canonical_seq,
            res_name_by_pos=resname_by_pos,
            coords_by_pos=coords_by_pos,
        )
    return chains


# ---------------------------------------------------------------------------
# Chain selection
# ---------------------------------------------------------------------------


def select_template_for_cif(
    query_seq: str,
    content: str,
    fmt: str,
    specified_chain_id: str | None,
    min_score: float = TEMPLATE_CIF_DIRECT_MIN_SCORE,
) -> SelectedTemplate | None:
    """Pick the best-aligning chain of one template CIF for a query sequence.

    Mirrors OSS ``CifDirectParser``: align each candidate chain to the query,
    score by seq_id * q_cov, and keep the best chain scoring >= ``min_score``.
    If ``specified_chain_id`` is given, only that chain is considered.
    """
    chains = extract_template_chains(content, fmt)
    if specified_chain_id is not None:
        candidates = {specified_chain_id: chains[specified_chain_id]} if specified_chain_id in chains else {}
        if not candidates:
            logger.warning(
                "Template chain_id %r not found in CIF (chains present: %s)", specified_chain_id, sorted(chains)
            )
    else:
        candidates = chains

    best: SelectedTemplate | None = None
    for chain_id, chain_data in candidates.items():
        idx_map, seq_id, q_cov = align_query_to_template_chain(query_seq, chain_data)
        score = seq_id * q_cov
        if score < min_score or idx_map.shape[0] == 0:
            continue
        if best is None or score > best.score:
            best = SelectedTemplate(chain_id=chain_id, idx_map=idx_map, score=score, chain_data=chain_data)
    return best


# ---------------------------------------------------------------------------
# Precursor filling (per query chain)
# ---------------------------------------------------------------------------


def resolve_template_idx_map(template: SelectedTemplate, chain_len: int) -> np.ndarray | None:
    """Replicate the OSS ``map_token_pos_to_template_residues`` keep/drop.

    Returns the effective ``idx_map`` (rows ``[query_res, template_pos]`` to
    write into the features) or ``None`` if the whole template is dropped.

    Two regimes for a protein query (one token per residue):

    * **All-standard template** — kept iff every query residue aligns to a real
      template residue (``idx_map`` covers the whole chain).
    * **Template with a modified residue** — every polymer amino-acid residue
      keeps its backbone after OSS ``add_unresolved_atoms``, so all aligned rows
      are kept (restype written; unresolved coords left NaN and masked).

    Deviation: a non-amino-acid polymer component lacking a protein backbone
    (dropped by OSS via its CCD composition check) is retained here — no CCD is
    consulted. Out of scope for protein template chains.
    """
    idx_map = template.idx_map  # rows aligned on both sides (real template res)
    chain_data = template.chain_data
    aligned_res_names = (chain_data.res_name_by_pos.get(int(t), "UNK") for t in np.unique(idx_map[:, 1]))
    nonstd = any(rn not in _STANDARD_PROTEIN_3 for rn in aligned_res_names)

    if not nonstd:
        # residue_starts (real aligned residues) vs repeats (== chain_len).
        return idx_map if idx_map.shape[0] == chain_len else None

    # Non-standard branch: every aligned polymer residue keeps its canonical
    # backbone after add_unresolved, so all are retained (restype written;
    # unresolved coordinates stay NaN and are masked).
    return idx_map


def fill_precursor_for_chain(
    template: SelectedTemplate,
    template_idx: int,
    idx_map: np.ndarray,
    token_pos_by_res_id: dict[int, int],
    res_names: np.ndarray,
    pseudo_beta_coords: np.ndarray,
    frame_coords: np.ndarray,
) -> None:
    """Write one template's residues into the precursor arrays in-place.

    For each aligned (query_res_id, template_pos) pair, place the residue name,
    pseudo-beta atom (CB, or CA for GLY), and N/CA/C frame at the query token
    position. Missing atoms remain NaN (masked downstream).
    """
    chain_data = template.chain_data
    for q_res, t_pos in idx_map:
        token_pos = token_pos_by_res_id.get(int(q_res))
        if token_pos is None:
            continue
        res_name = chain_data.res_name_by_pos.get(int(t_pos))
        if res_name is None:
            continue
        res_names[template_idx, token_pos] = res_name

        atoms = chain_data.coords_by_pos.get(int(t_pos), {})
        # Pseudo-beta: CA for glycine, CB otherwise.
        pb_name = "CA" if res_name == "GLY" else "CB"
        if pb_name in atoms:
            pseudo_beta_coords[template_idx, token_pos, :] = atoms[pb_name]
        for k, aname in enumerate(_FRAME_ATOMS):
            if aname in atoms:
                frame_coords[template_idx, token_pos, k, :] = atoms[aname]
