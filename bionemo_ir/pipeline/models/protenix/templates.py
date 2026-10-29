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

"""Supplied-template alignment and Protenix template geometry."""

import numpy as np
import torch

from bionemo_ir.pipeline.base import FeatureGeneratorBase
from bionemo_ir.pipeline.models.openfold3.common import create_template_unit_vector
from bionemo_ir.pipeline.models.openfold3.template_logic import (
    SelectedTemplate,
    TemplateParsing,
    select_template_for_cif,
)
from bionemo_ir.pipeline.utils.atom import ATOM_NAME_TO_FOLDING_INDEX
from bionemo_ir.pipeline.utils.template import masked_distogram

from .const import PROTEIN_CODES, RES_ATOMS_DICT, RESTYPES
from .feature_generators import structure_context

_DENSE_ATOMS = 24
# The release parses templates with Biopython: all heavy atoms, standard 3-to-1 map, per-atom altlocs.
_RELEASE_PARSING = TemplateParsing(atom_names=None, modified_as_unknown=True, altloc="atom")
_PROTEIN_ATOMS = frozenset().union(*(RES_ATOMS_DICT[name] for name in list(RESTYPES)[:20]))


def _geometry_slots() -> tuple[np.ndarray, np.ndarray]:
    """Per-restype dense slots of the pseudo-beta and (C, CA, N) frame atoms; -1 when absent.

    ``UNK`` and gap fall back to slot 0 for whichever of the two the residue lacks.
    """
    beta = np.full(len(RESTYPES), -1, dtype=np.int64)
    frame = np.full((len(RESTYPES), 3), -1, dtype=np.int64)
    for name, index in RESTYPES.items():
        atoms = RES_ATOMS_DICT.get(name, {})
        fallback = 0 if name in {"UNK", "-"} else -1
        beta[index] = atoms.get("CA" if name == "GLY" else "CB", fallback)
        if all(label in atoms for label in ("C", "CA", "N")):
            frame[index] = [atoms[label] for label in ("C", "CA", "N")]
        else:
            frame[index] = fallback
    return beta, frame


_BETA_SLOT, _FRAME_SLOTS = _geometry_slots()
_DISTOGRAM_LOWER = np.linspace(3.25, 50.75, 39, dtype=np.float32) ** 2
# Protenix templates do not mask cross-chain pairs.
_ALL_PAIRS = torch.ones((1, 1, 1, 1))


def _gather(values: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Pick ``values[t, n, slot]`` along the atom axis, zero where ``slot < 0``."""
    shape = slots.shape + (1,) * (values.ndim - slots.ndim)
    return np.take_along_axis(values, np.maximum(slots, 0).reshape(shape), axis=2) * (slots >= 0).reshape(shape)


def template_geometry(restypes: np.ndarray, positions: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
    """Compute pair features from dense template atom coordinates.

    Slots without resolved atoms stay zero, so absent templates cost no pair work.

    Args:
        restypes: ``[T, N]`` int64 template residue types.
        positions: ``[T, N, 24, 3]`` float32 dense atom coordinates.
        mask: ``[T, N, 24]`` float32 atom presence.
    """
    count, n = restypes.shape
    distogram = np.zeros((count, n, n, len(_DISTOGRAM_LOWER)), dtype=np.float32)
    pair_mask = np.zeros((count, n, n), dtype=np.float32)
    unit_vector = np.zeros((count, n, n, 3), dtype=np.float32)
    frame_mask = np.zeros((count, n, n), dtype=np.float32)
    for t in np.flatnonzero(mask.any(axis=(1, 2))):
        slot = slice(t, t + 1)
        beta_slot, frame_slots = _BETA_SLOT[restypes[slot]], _FRAME_SLOTS[restypes[slot]]
        beta = _gather(positions[slot], beta_slot[..., None])[0, :, 0]
        beta_mask = _gather(mask[slot], beta_slot[..., None])[0, :, 0]
        backbone = _gather(positions[slot], frame_slots)[0]
        backbone_mask = _gather(mask[slot], frame_slots).prod(-1)[0]
        beta *= beta_mask[:, None]
        pair_mask[t] = beta_mask[:, None] * beta_mask[None, :]
        masked_distogram(beta[None], beta_mask[None], _DISTOGRAM_LOWER, 1e8, output=distogram[slot])
        frame_mask[t] = backbone_mask[:, None] * backbone_mask[None, :]
        # Frame slots hold (C, CA, N); the shared helper takes (N, CA, C).
        unit_vector[t] = create_template_unit_vector(
            np.ascontiguousarray(backbone[None, :, ::-1]), torch.from_numpy(backbone_mask[None]), _ALL_PAIRS
        )[0].numpy()
    return {
        "template_aatype": torch.from_numpy(restypes),
        "template_distogram": torch.from_numpy(distogram),
        "template_pseudo_beta_mask": torch.from_numpy(pair_mask),
        "template_unit_vector": torch.from_numpy(unit_vector),
        "template_backbone_frame_mask": torch.from_numpy(frame_mask),
    }


def _resolved_atoms(template: SelectedTemplate) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    """Finite protein heavy atoms per template position, with MSE selenium named SD, and their center.

    Release templates center the full resolved protein chain before alignment.
    """
    chain = template.chain_data
    atoms = {}
    for position, coords in chain.coords_by_pos.items():
        selenomethionine = chain.res_name_by_pos.get(position) == "MSE"
        atoms[position] = {
            label: np.asarray(xyz, dtype=np.float32)
            for name, xyz in coords.items()
            if (label := "SD" if selenomethionine and name == "SE" else name) in _PROTEIN_ATOMS
            and np.isfinite(xyz).all()
        }
    for position, arg in atoms.items():
        if chain.res_name_by_pos.get(position) == "ARG" and all(k in arg for k in ("CD", "NH1", "NH2")):
            if np.linalg.norm(arg["NH1"] - arg["CD"]) > np.linalg.norm(arg["NH2"] - arg["CD"]):
                arg["NH1"], arg["NH2"] = arg["NH2"], arg["NH1"]
    # Sum in the release's atom37 layout order so the float32 mean matches.
    present = [
        atoms[position][label]
        for position in sorted(atoms)
        for label in sorted(atoms[position], key=ATOM_NAME_TO_FOLDING_INDEX.__getitem__)
    ]
    center = np.asarray(present, dtype=np.float32).mean(0) if present else np.zeros(3)
    return atoms, center


def _template_hits(sequence: str, templates: list[dict], count: int) -> list[tuple[SelectedTemplate, dict, np.ndarray]]:
    """Aligned supplied templates in supplied order, with their resolved atoms and center."""
    selected = []
    for template in templates:
        if template.get("content") is None:
            raise ValueError("Protenix template content missing")
        if (template.get("format") or "cif").lower() != "cif":
            raise ValueError("Protenix templates must be mmCIF")
        hit = select_template_for_cif(
            sequence,
            template["content"],
            "cif",
            template.get("chain_id"),
            # Caller-supplied templates skip search-hit filtering, as in the release runner.
            min_score=0.0,
            parsing=_RELEASE_PARSING,
        )
        if hit is not None:
            selected.append(hit)
    # The release consumes hits in file order, so supplied order ranks the slots.
    return [(hit, *_resolved_atoms(hit)) for hit in selected[:count]]


class TemplateFeatureGenerator(FeatureGeneratorBase):
    """Align supplied protein templates and retain their resolved atoms."""

    def __call__(self, batch: dict, context: dict) -> dict:
        structure = structure_context(context)
        atoms, tokens = structure["atoms"], structure["tokens"]
        n, count = len(tokens), self.config.max_templates
        restypes = np.zeros((count, n), dtype=np.int64)
        restypes[0] = 31
        positions = np.zeros((count, n, _DENSE_ATOMS, 3), dtype=np.float32)
        masks = np.zeros((count, n, _DENSE_ATOMS), dtype=np.float32)
        if self.config.use_templates:
            # Homomer copies share one parse and alignment per distinct sequence and template list.
            hits: dict[tuple, list] = {}
            for aid, chain in enumerate(structure["chains"]):
                templates = chain.get("templates") or []
                if chain["polymer_type"] != "protein" or len(chain["sequence"]) <= 4 or not templates:
                    continue
                key = (
                    chain["sequence"],
                    tuple((t.get("content"), t.get("format"), t.get("chain_id")) for t in templates),
                )
                if key not in hits:
                    hits[key] = _template_hits(chain["sequence"], templates, count)
                residue_tokens = {
                    int(atoms.res_id[t["center"]]): i for i, t in enumerate(tokens) if atoms.asym[t["center"]] == aid
                }
                for tid, (template, resolved, center) in enumerate(hits[key]):
                    restypes[tid, list(residue_tokens.values())] = 31
                    for query_res, template_res in template.idx_map:
                        token = residue_tokens.get(int(query_res))
                        if token is None:
                            continue
                        name = PROTEIN_CODES.get(template.chain_data.canonical_seq[int(template_res) - 1], "UNK")
                        restypes[tid, token] = RESTYPES.get(name, 20)
                        coords = resolved.get(int(template_res), {})
                        labels = [a for a in RES_ATOMS_DICT.get(name, {}) if a != "OXT"] if name != "UNK" else []
                        # Dense padding maps to atom37 slot zero (N) in the release.
                        labels += ["N"] * (_DENSE_ATOMS - len(labels))
                        for idx, label in enumerate(labels):
                            if label in coords:
                                positions[tid, token, idx] = coords[label] - center
                                masks[tid, token, idx] = 1
        return template_geometry(restypes, positions, masks)
