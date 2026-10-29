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

"""Native release-compatible MSA assembly and species pairing."""

import re
import string

import numpy as np
import torch
import torch.nn.functional as F

from bionemo_ir.pipeline.base import FeatureGeneratorBase
from bionemo_ir.pipeline.utils.msa import a3m_columns, map_code_points, msa_profile

from .const import MSA_DNA_SEQ_TO_ID, MSA_PROTEIN_SEQ_TO_ID, MSA_RNA_SEQ_TO_ID
from .feature_generators import structure_context, token_restypes

_SPECIES = re.compile(r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|[A-Z0-9]{1,10}_([A-Z0-9]{1,5})|^UniRef100_[^_]+_([^_/]+)")
_DELETE_LOWERCASE = str.maketrans("", "", string.ascii_lowercase)


def _lookup(mapping: dict[str, int]) -> np.ndarray:
    table = np.full(128, -1, dtype=np.int32)
    for char, value in mapping.items():
        table[ord(char)] = value
    return table


_TABLES = {
    "protein": _lookup(MSA_PROTEIN_SEQ_TO_ID),
    "rna": _lookup(MSA_RNA_SEQ_TO_ID),
    "dna": _lookup(MSA_DNA_SEQ_TO_ID),
}


def _invalid_residue(char: str) -> int:
    raise ValueError(f"Invalid Protenix MSA residue: {char!r}")


def alignment_rows(chain: dict, paired: bool, pair_as_unpaired: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode parsed alignments, preserving deletion counts and species."""
    kind, query = chain["polymer_type"], chain["sequence"]
    if kind not in _TABLES:
        query, kind = "X" * chain["msa_length"], "protein"
    entries = chain.get("paired_msas" if paired else "msas") or []
    if not paired and pair_as_unpaired:
        entries = (chain.get("paired_msas") or []) + entries
    if len(query) <= 4 or kind == "dna":
        entries = []
    sequences, descriptions = [], []
    for entry in entries:
        raw = entry.get("raw") or entry.get("sequences") or []
        labels = entry.get("descriptions") or [""] * len(raw)
        if len(labels) != len(raw):
            raise ValueError("Protenix MSA descriptions must match the sequence count")
        sequences.extend(raw)
        descriptions.extend(labels)
    if not sequences:
        sequences, descriptions = [query], [""]
    if not paired:
        first: dict[str, int] = {}
        for index, sequence in enumerate(sequences):
            first.setdefault(sequence.translate(_DELETE_LOWERCASE), index)
        selected = list(first.values())
        sequences = [sequences[i] for i in selected]
        descriptions = [descriptions[i] for i in selected]
    codes, deletions, counts = a3m_columns(sequences)
    if np.any(counts != len(query)):
        raise ValueError("Protenix MSA aligned row length must match the query")
    rows = map_code_points(codes, _TABLES[kind], _invalid_residue).reshape(len(sequences), len(query))
    deletions = deletions.astype(np.int32).reshape(rows.shape)
    species = []
    for description in descriptions:
        match = _SPECIES.match(description.strip())
        species.append(next((g for g in match.groups() if g), "") if match else "")
    return rows, deletions, species


def paired_indices(species: list[list[str]], limit: int, per_species: int) -> np.ndarray:
    """Rank species blocks by chain coverage and source-row indices."""
    maps, counts, depths = [], {}, {}
    for ids in species:
        identifiers = np.asarray(ids)
        order = identifiers.argsort()
        names, starts = np.unique(identifiers[order], return_index=True)
        groups = dict(zip(names, np.split(order, starts[1:]), strict=True))
        maps.append(groups)
        for name, rows in groups.items():
            counts[name] = counts.get(name, 0) + 1
            depths[name] = min(depths.get(name, per_species), len(rows))
    selected = [np.zeros((1, len(maps)), dtype=np.int64)]
    covered = 0
    for coverage in sorted(set(counts.values()), reverse=True):
        if coverage < 2:
            continue
        blocks = []
        for name, count in counts.items():
            if not name or count != coverage:
                continue
            depth = depths[name]
            blocks.append(np.stack([m.get(name, np.full(depth, -1))[:depth] for m in maps], axis=1))
        if blocks:
            block = np.concatenate(blocks)
            rank = np.log(np.abs(block.astype(np.float32)) + 1e-10).sum(1)
            selected.append(block[np.argsort(rank)])
            covered += len(block)
        if covered >= limit:
            break
    return np.concatenate(selected)[:limit]


class MsaFeatureGenerator(FeatureGeneratorBase):
    """Assemble protein/RNA alignments and project them onto tokens."""

    def __call__(self, batch: dict, context: dict) -> dict:
        structure = structure_context(context)
        if not self.config.use_msa:
            query = token_restypes(structure)
            n = len(query)
            return {
                "msa": query[None],
                "has_deletion": torch.zeros(1, n),
                "deletion_value": torch.zeros(1, n),
                "profile": F.one_hot(query, 32).float(),
                "deletion_mean": torch.zeros(n),
            }
        atoms, tokens = structure["atoms"], structure["tokens"]
        chains = [dict(c) for c in structure["chains"]]
        need_pair = len({c["sequence"] for c in chains if c["polymer_type"] == "protein"}) > 1
        unpaired, paired, species, profiles, means = [], [], [], [], []
        for aid, chain in enumerate(chains):
            chain["msa_length"] = int((atoms.asym == aid).sum())
            if chain["polymer_type"] == "rna" and not self.config.use_rna_msa:
                chain["msas"] = []
                chain["paired_msas"] = []
            u, ud, _ = alignment_rows(chain, False, self.config.msa_pair_as_unpair)
            if not need_pair or chain["polymer_type"] != "protein":
                chain["paired_msas"] = []
            p, pd, sp = alignment_rows(chain, True, False)
            profiles.append(msa_profile(torch.from_numpy(u), 32).numpy())
            means.append(ud.mean(0).astype(np.float32))
            unpaired.append((u, ud))
            paired.append((p, pd))
            species.append(sp)
        limit = self.config.max_msa_rows
        pair_limit = max(1, limit // 2)
        if need_pair:
            selection = paired_indices(species, pair_limit, self.config.max_paired_per_species)
            for chain, indices in enumerate(selection.T):
                paired[chain] = tuple(
                    np.concatenate([v, np.full((1, v.shape[1]), fill, dtype=v.dtype)])[indices]
                    for v, fill in zip(paired[chain], (31, 0), strict=True)
                )
                used = {r.astype(np.int8).tobytes() for r in paired[chain][0]}
                keep = [i for i, row in enumerate(unpaired[chain][0]) if row.astype(np.int8).tobytes() not in used]
                unpaired[chain] = tuple(v[keep] for v in unpaired[chain])
        keep_pair = np.any(np.concatenate([p[0] for p in paired], axis=1) != 31, axis=1)
        paired = [tuple(v[keep_pair][:pair_limit] for v in p) for p in paired]
        unpaired = [tuple(v[: max(0, limit - len(p[0]))] for v in u) for p, u in zip(paired, unpaired, strict=True)]
        depth = max(len(u[0]) for u in unpaired)
        up = [
            np.concatenate(
                [np.pad(u[k], ((0, depth - len(u[k])), (0, 0)), constant_values=fill) for u in unpaired], axis=1
            )
            for k, fill in enumerate((31, 0))
        ]
        pa = [np.concatenate([p[k] for p in paired], axis=1) for k in (0, 1)]
        rows, deletion = [np.concatenate([pa[k], up[k]]) for k in (0, 1)]
        offsets = np.cumsum([0] + [u[0].shape[1] for u in unpaired])
        for aid, chain in enumerate(chains):
            if chain["polymer_type"] != "protein":
                columns = slice(offsets[aid], offsets[aid + 1])
                gaps = np.all(rows[:, columns] == 31, axis=1)
                rows[gaps, columns] = rows[0, columns]
        centers = np.array([t["center"] for t in tokens])
        columns = offsets[atoms.asym[centers]] + atoms.res_id[centers] - 1
        rows = rows[:, columns].astype(np.int64)
        deletion = np.clip(deletion[:, columns], -128, 127).astype(np.int8)
        return {
            "msa": torch.from_numpy(rows),
            "has_deletion": torch.from_numpy(np.clip(deletion, 0.0, 1.0).astype(np.float32)),
            "deletion_value": torch.from_numpy((np.arctan(deletion / 3.0) * (2.0 / np.pi)).astype(np.float32)),
            "profile": torch.from_numpy(np.concatenate(profiles)[columns]),
            "deletion_mean": torch.from_numpy(np.concatenate(means)[columns]),
        }
