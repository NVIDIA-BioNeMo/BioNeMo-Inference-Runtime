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

"""Boltz1 feature computation from Structure + Tokens.

Differences from Boltz2 (featurizerv2):
- MSA is one-hot (N_MSA, L, 33) instead of token indices (N_MSA, L)
- frames_idx / frame_resolved_mask have no ensemble dim: (L, 3) / (L,)
- Has pocket_feature, no template/contact/method/affinity/backbone features
- ref_atom_name_chars uses % num_bins encoding
- ref_charge is int8 not float32
- atom_to_token is one-hot int64 (same as Boltz2 but explicit)
- No token_to_center_atom, no ref_chirality, no bfactor/plddt atom features
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.functional import one_hot

# isort: off
from bionemo_ir.pipeline.models.boltz2.const import (
    Structure,
    Token,
    TokenBond,
    chain_type_ids,
    num_elements,
    num_tokens,
    ref_atoms,
)
from bionemo_ir._torch.utils import pad_dim
from bionemo_ir.pipeline.models.boltz2.featurizer import (
    _center_random_augmentation,
    _compute_collinear_mask,
    _convert_atom_name,
    _fill_nonpolymer_frames,
    _frame_resolved_mask_oss,
)
# isort: on

# Boltz1 pocket contact info (from OSS boltz.data.const)
pocket_contact_info = {
    "UNSPECIFIED": 0,
    "BINDER": 1,
    "POCKET": 2,
    "UNSELECTED": 3,
}


def process_token_features(
    tokens_list: list[Token],
    token_bonds: list[TokenBond],
    structure: Structure,
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build Boltz1 token-level feature tensors."""
    n = len(tokens_list)
    num_tok = max_tokens if max_tokens is not None else n
    pad_len = (num_tok - n) if max_tokens is not None and n < max_tokens else 0

    token_index = torch.arange(n, dtype=torch.long)
    residue_index = torch.tensor([t.res_idx for t in tokens_list], dtype=torch.long)
    asym_id = torch.tensor([t.asym_id for t in tokens_list], dtype=torch.long)
    entity_id = torch.tensor([t.entity_id for t in tokens_list], dtype=torch.long)
    sym_id = torch.tensor([t.sym_id for t in tokens_list], dtype=torch.long)
    mol_type = torch.tensor([t.mol_type for t in tokens_list], dtype=torch.long)
    res_type = torch.tensor([t.res_type for t in tokens_list], dtype=torch.long)
    res_type = one_hot(res_type, num_classes=num_tokens)
    disto_center = torch.tensor([t.disto_coords for t in tokens_list], dtype=torch.float32)
    cyclic_period = torch.tensor([t.cyclic_period for t in tokens_list], dtype=torch.long)

    pad_mask = torch.ones(n, dtype=torch.float32)
    resolved_mask = torch.tensor([t.resolved_mask for t in tokens_list], dtype=torch.float32)
    disto_mask = torch.tensor([t.disto_mask for t in tokens_list], dtype=torch.float32)

    tok_to_idx = {t.token_idx: i for i, t in enumerate(tokens_list)}
    bonds = torch.zeros(num_tok, num_tok, dtype=torch.float32)
    for tb in token_bonds:
        i1 = tok_to_idx.get(tb.token_1)
        i2 = tok_to_idx.get(tb.token_2)
        if i1 is not None and i2 is not None:
            bonds[i1, i2] = 1
            bonds[i2, i1] = 1
    bonds = bonds.unsqueeze(-1)

    # Pocket feature: default UNSPECIFIED (inference, no binder/pocket info)
    pocket_np = np.full(n, pocket_contact_info["UNSPECIFIED"], dtype=np.int64)
    pocket_feature = torch.from_numpy(pocket_np).long()
    pocket_feature = one_hot(pocket_feature, num_classes=len(pocket_contact_info))

    if pad_len > 0:
        token_index = pad_dim(token_index, 0, pad_len)
        residue_index = pad_dim(residue_index, 0, pad_len)
        asym_id = pad_dim(asym_id, 0, pad_len)
        entity_id = pad_dim(entity_id, 0, pad_len)
        sym_id = pad_dim(sym_id, 0, pad_len)
        mol_type = pad_dim(mol_type, 0, pad_len)
        res_type = pad_dim(res_type, 0, pad_len)
        disto_center = pad_dim(disto_center, 0, pad_len)
        cyclic_period = pad_dim(cyclic_period, 0, pad_len)
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        disto_mask = pad_dim(disto_mask, 0, pad_len)
        pocket_feature = pad_dim(pocket_feature, 0, pad_len)

    return {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": bonds,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "pocket_feature": pocket_feature,
        "cyclic_period": cyclic_period,
    }


def process_atom_features(
    structure: Structure,
    tokens_list: list[Token],
    molecules: dict[str, Any],
    num_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    atoms_per_window_queries: int = 32,
    max_atoms: int | None = None,
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build Boltz1 atom-level feature tensors (no ensemble dim)."""
    n_tokens = len(tokens_list)
    atom_to_token = []
    token_to_rep_atom = []
    ref_space_uid = []
    atom_name_list = []
    atom_element_list = []
    atom_charge_list = []
    atom_conformer_list = []
    frame_data = []
    resolved_frame_data = []
    coord_data_list = []
    atom_idx = 0
    chain_res_ids: dict = {}

    for token_id, token in enumerate(tokens_list):
        chain_idx, res_id = token.asym_id, token.res_idx
        key = (chain_idx, res_id)
        if key not in chain_res_ids:
            chain_res_ids[key] = len(chain_res_ids)
        new_idx = chain_res_ids[key]

        start = token.atom_idx
        end = token.atom_idx + token.atom_num
        token_atoms = structure.atoms[start:end]

        ref_space_uid.extend([new_idx] * token.atom_num)
        atom_to_token.extend([token_id] * token.atom_num)

        # OSS boltz1 process_atom_features reads ref_element / ref_charge /
        # ref_pos (conformer) straight from structure.atoms — these were
        # populated from the CCD ref_mol during parsing (the same ccd.pkl mols
        # boltz1 inference uses). Do NOT re-load a mol_dir molecule here:
        # mol_dir mols carry a different conformer set, so ref_pos would
        # diverge from the reference implementation.
        for a in token_atoms:
            atom_name_list.append(_convert_atom_name(a.name))
            atom_element_list.append(a.element)
            atom_charge_list.append(a.charge)
            atom_conformer_list.append(a.conformer)

        token_to_rep_atom.append(atom_idx + token.disto_idx - start)
        chain = structure.chains[token.asym_id]

        # Coord data: single conformer (no ensemble)
        token_coords = np.array(structure.coords[start:end], dtype=np.float32)
        coord_data_list.append(token_coords)

        # Frame data (matches OSS boltz1 featurizer.py process_atom_features):
        # protein -> N/CA/C; RNA/DNA -> C1'/C3'/C4'; NONPOLYMER frames are
        # rebuilt below by _fill_nonpolymer_frames.
        if token.atom_num >= 3 and token.res_name in ref_atoms and ref_atoms[token.res_name][:3] == ["N", "CA", "C"]:
            frame_data.append([start, start + 1, start + 2])
        elif (
            token.atom_num >= 3
            and token.res_name in ref_atoms
            and chain.mol_type in (chain_type_ids["DNA"], chain_type_ids["RNA"])
        ):
            try:
                ref = ref_atoms[token.res_name]
                idx_c1 = ref.index("C1'")
                idx_c3 = ref.index("C3'")
                idx_c4 = ref.index("C4'")
                frame_data.append([start + idx_c1, start + idx_c3, start + idx_c4])
            except ValueError:
                frame_data.append([token.center_idx, token.center_idx, token.center_idx])
        else:
            frame_data.append([token.center_idx, token.center_idx, token.center_idx])
        resolved_frame_data.append(
            _frame_resolved_mask_oss(token_atoms, chain.mol_type, token.res_name, token.atom_num, token.res_type)
        )
        atom_idx += token.atom_num

    # Coord data: (1, n_atoms, 3) single ensemble
    coord_data = np.stack(coord_data_list) if len(coord_data_list) == 1 else np.concatenate(coord_data_list, axis=0)
    coord_data = coord_data.reshape(1, -1, 3)

    # Frame collinear mask
    frame_data_arr = np.array(frame_data, dtype=np.int64)

    # OSS boltz1 compute_frames_nonpolymer: for each NONPOLYMER chain with >=3
    # atoms, replace the per-atom token frame with (nearest_1, self, nearest_2)
    # from intra-chain pairwise distances (resolved atoms preferred). Identical
    # algorithm to boltz2 _fill_nonpolymer_frames; coord_data is (1, n_atoms, 3).
    _fill_nonpolymer_frames(
        tokens=tokens_list,
        structure=structure,
        coord_data=coord_data,
        frame_data_arr=frame_data_arr,
        resolved_frame_data=resolved_frame_data,
    )

    frames_expanded = coord_data[0][frame_data_arr]
    v1 = frames_expanded[:, 1] - frames_expanded[:, 0]
    v2 = frames_expanded[:, 1] - frames_expanded[:, 2]
    mask_collinear = _compute_collinear_mask(v1, v2)
    resolved_frame_np = np.array([float(x) for x in resolved_frame_data], dtype=np.float32)
    resolved_frame_np = resolved_frame_np * mask_collinear

    # Build tensors
    atom_name_arr = np.array(atom_name_list, dtype=np.int32)
    atom_element_arr = np.array(atom_element_list, dtype=np.int64)
    atom_charge_arr = np.array(atom_charge_list, dtype=np.int8)  # Boltz1: int8
    atom_conformer_arr = np.array(atom_conformer_list, dtype=np.float32)
    ref_space_uid_arr = np.array(ref_space_uid, dtype=np.int64)

    resolved_mask = torch.tensor(
        [structure.atoms[i].is_present for i in range(atom_idx)], dtype=torch.bool
    )  # Boltz1: bool
    pad_mask = torch.ones(atom_idx, dtype=torch.float32)
    atom_to_token_t = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom_t = torch.tensor(token_to_rep_atom, dtype=torch.long)

    ref_pos = torch.from_numpy(atom_conformer_arr).float()
    # Boltz1: ref_atom_name_chars uses % num_bins (upstream
    # ``process_atom_features`` in ``boltz/data/feature/featurizer.py``)
    ref_atom_name_chars = torch.from_numpy(atom_name_arr).long()
    ref_element = torch.from_numpy(atom_element_arr).long()
    ref_charge = torch.from_numpy(atom_charge_arr)  # int8
    ref_space_uid_t = torch.from_numpy(ref_space_uid_arr).long()
    coords = torch.from_numpy(coord_data).float()

    # Upstream Boltz1 ``process_atom_features``: whole-tensor augmentation
    # (not per-ref-space)
    resolved_mask_f = resolved_mask.float()
    ref_pos = _center_random_augmentation(
        ref_pos[None],
        resolved_mask_f[None],
        s_trans=1.0,
        augmentation=True,
        centering=False,
    )[0]

    # Center ground-truth coords by resolved_mask
    center = (coords * resolved_mask_f[None, :, None]).sum(dim=1)
    center = center / resolved_mask_f.sum().clamp(min=1)
    coords = coords - center[:, None]

    # One-hot encodings: Boltz1 uses % num_bins for atom names
    ref_atom_name_chars = one_hot(ref_atom_name_chars % num_bins, num_classes=num_bins)
    ref_element = one_hot(ref_element, num_classes=num_elements)
    atom_to_token_t = one_hot(atom_to_token_t, num_classes=n_tokens)
    token_to_rep_atom_t = one_hot(token_to_rep_atom_t, num_classes=atom_idx)

    # Pad atoms
    pad_len_atom = ((atom_idx - 1) // atoms_per_window_queries + 1) * atoms_per_window_queries - atom_idx
    if max_atoms is not None:
        pad_len_atom = max_atoms - atom_idx
    if pad_len_atom > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len_atom)
        ref_pos = pad_dim(ref_pos, 0, pad_len_atom)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len_atom)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len_atom)
        ref_element = pad_dim(ref_element, 0, pad_len_atom)
        ref_charge = pad_dim(ref_charge, 0, pad_len_atom)
        ref_space_uid_t = pad_dim(ref_space_uid_t, 0, pad_len_atom)
        coords = pad_dim(coords, 1, pad_len_atom)
        atom_to_token_t = pad_dim(atom_to_token_t, 0, pad_len_atom)
        token_to_rep_atom_t = pad_dim(token_to_rep_atom_t, 1, pad_len_atom)

    # Boltz1: frames have no ensemble dim. Use frame_data_arr (NOT the raw
    # frame_data list) so the NONPOLYMER frames rebuilt in-place by
    # _fill_nonpolymer_frames above are reflected here.
    frames_idx = torch.from_numpy(frame_data_arr).long()
    frame_resolved_mask = torch.from_numpy(resolved_frame_np).bool()

    if max_tokens is not None and n_tokens < max_tokens:
        pl = max_tokens - n_tokens
        frames_idx = pad_dim(frames_idx, 0, pl)
        frame_resolved_mask = pad_dim(frame_resolved_mask, 0, pl)
        atom_to_token_t = pad_dim(atom_to_token_t, 1, pl)
        token_to_rep_atom_t = pad_dim(token_to_rep_atom_t, 0, pl)

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_space_uid": ref_space_uid_t,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token_t,
        "token_to_rep_atom": token_to_rep_atom_t,
        "frames_idx": frames_idx,
        "frame_resolved_mask": frame_resolved_mask,
    }


def process_msa_features(
    tokens_list: list[Token],
    msa_parsed_per_chain: list[Any | None] | None = None,
    paired_msa_per_chain: list[Any | None] | None = None,
    max_seqs: int = 16384,
    max_paired: int = 8192,
    max_tokens: int | None = None,
    pad_to_max_seqs: bool = False,
) -> dict[str, torch.Tensor]:
    """Build Boltz1 MSA features: msa is one-hot (N_MSA, L, 33)."""
    from bionemo_ir.pipeline.models.boltz2.featurizer import process_msa_features as _boltz2_msa

    # Reuse Boltz2 MSA construction (produces token indices)
    b2 = _boltz2_msa(
        tokens_list, msa_parsed_per_chain, paired_msa_per_chain, max_seqs, max_paired, max_tokens, pad_to_max_seqs
    )

    # Boltz1: msa is one-hot (N_MSA, L, 33), msa_mask is int64
    msa_indices = b2["msa"]  # (N_MSA, L) long
    msa_onehot = one_hot(msa_indices, num_classes=num_tokens)  # (N_MSA, L, 33)
    msa_mask = b2["msa_mask"].long()  # int64
    has_deletion = b2["has_deletion"]  # bool

    return {
        "msa": msa_onehot,
        "msa_paired": b2["msa_paired"],
        "deletion_value": b2["deletion_value"],
        "has_deletion": has_deletion,
        "deletion_mean": b2["deletion_mean"],
        "profile": b2["profile"],
        "msa_mask": msa_mask,
    }
