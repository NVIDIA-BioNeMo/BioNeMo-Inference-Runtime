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

"""Select Protenix's highest-ranked candidate for the shared structure writers."""

import numpy as np

from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.data.schemas.basic import MOL_TYPE_DNA, MOL_TYPE_LIGAND, MOL_TYPE_PROTEIN, MOL_TYPE_RNA
from bionemo_ir.pipeline.base import PostProcessorBase
from bionemo_ir.pipeline.utils.atom import (
    decode_atom_name_chars,
    decode_element_symbols,
    one_hot_indices,
    scatter_flat_atoms_to_folding_layout,
    to_numpy,
)

# Indexed by the model's mol-type ids (MOL_TYPES in const.py).
_FOLDING_MOL_TYPES = np.array([MOL_TYPE_PROTEIN, MOL_TYPE_DNA, MOL_TYPE_RNA, MOL_TYPE_LIGAND])
# Batch keys read below; a consuming forward must leave them in place.
INPUT_FEATURES = (
    "atom_mask",
    "token_mask",
    "atom_to_token_idx",
    "ref_atom_name_chars",
    "ref_element",
    "restype",
    "residue_index",
    "asym_id",
    "token_resnames",
    "token_mol_types",
)


class PostProcessor(PostProcessorBase):
    def __call__(self, batch: dict, output: dict) -> FoldingOutput:
        summaries, full = output["summary_confidence"], output["full_data"]
        best = max(range(len(summaries)), key=lambda i: float(summaries[i]["ranking_score"]))
        coordinates = output["coordinate"]
        if coordinates.ndim == 4:
            coordinates = coordinates[0]
        active = to_numpy(batch["atom_mask"]).reshape(-1).astype(bool)
        ownership = to_numpy(batch["atom_to_token_idx"]).reshape(-1).astype(np.int64)
        n = int(to_numpy(batch["token_mask"]).sum())
        names = decode_atom_name_chars(batch["ref_atom_name_chars"][0], active)
        # ref_element one-hots atomic number minus one.
        elements = decode_element_symbols(batch["ref_element"][0], active, index_offset=1)
        scattered = scatter_flat_atoms_to_folding_layout(
            to_numpy(coordinates[best]), ownership, names, active, n, atom_elements=elements
        )
        atom_plddt = to_numpy(full[best]["atom_plddt"]) * 100
        # Same scatter, so each atom's pLDDT lands in its coordinate slot.
        b_factors = scatter_flat_atoms_to_folding_layout(
            np.repeat(atom_plddt[:, None], 3, axis=1), ownership, names, active, n
        ).positions[..., 0]
        counts = np.bincount(ownership[active], minlength=n)
        plddt = np.bincount(ownership[active], weights=atom_plddt[active], minlength=n) / np.maximum(counts, 1)
        pae = to_numpy(full[best]["token_pair_pae"])[:n, :n]
        return FoldingOutput(
            atom_positions=scattered.positions,
            atom_mask=scattered.mask,
            residue_types=one_hot_indices(batch["restype"][0])[:n],
            residue_indices=to_numpy(batch["residue_index"]).reshape(-1)[:n],
            chain_indices=to_numpy(batch["asym_id"]).reshape(-1)[:n],
            residue_names=list(batch["token_resnames"][:n]),
            mol_types=_FOLDING_MOL_TYPES[np.asarray(batch["token_mol_types"][:n], dtype=np.int64)],
            b_factors=b_factors,
            plddt=plddt.astype(np.float32),
            ptm=float(summaries[best]["ptm"]),
            iptm=float(summaries[best]["iptm"]),
            pae=pae,
            max_pae=float(pae.max()),
            atom_names=scattered.names,
            atom_elements=scattered.elements,
        )
