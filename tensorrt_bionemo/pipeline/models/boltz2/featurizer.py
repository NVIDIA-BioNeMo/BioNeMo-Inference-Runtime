# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boltz2 feature computation from Structure + Tokens (no OSS imports)."""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch
from torch.nn.functional import one_hot

from tensorrt_bionemo._torch.layers.random_augmentation import random_rotations
from tensorrt_bionemo._torch.tensor_utils import pad_dim

from .const import (Structure, Token, TokenBond, chain_type_ids,
                    contact_conditioning_info, method_types_ids,
                    nucleic_backbone_atom_index, num_elements, num_tokens,
                    protein_backbone_atom_index, ref_atoms, token_ids, tokens)


def _center_random_augmentation(
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
) -> torch.Tensor:
    """OSS Algorithm 19: center (optional) then random rotation + translation. Deterministic under seed."""
    if centering:
        atom_mean = (
            (atom_coords * atom_mask[:, :, None]).sum(dim=1, keepdim=True) /
            (atom_mask[:, :, None].sum(dim=1, keepdim=True).clamp(min=1e-8)))
        atom_coords = atom_coords - atom_mean
    if augmentation:
        R = random_rotations(
            atom_coords.shape[0],
            dtype=atom_coords.dtype,
            device=atom_coords.device,
        )
        atom_coords = torch.einsum("bmd,bds->bms", atom_coords, R)
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans
    return atom_coords


def _frame_resolved_mask_oss(
    token_atoms: list,
    mol_type: int,
    res_name: str,
    atom_num: int,
    res_type: int,
) -> bool:
    """Compute frame resolved mask per token (OSS featurizerv2 lines 1280-1368)."""
    res_type_name = tokens[res_type] if res_type < len(tokens) else "UNK"
    if atom_num < 3 or res_type_name in ["PAD", "UNK", "-"]:
        return False
    names = [a.name for a in token_atoms]
    present = [a.is_present for a in token_atoms]
    if mol_type == chain_type_ids["PROTEIN"] and res_name in ref_atoms:
        try:
            ref = ref_atoms[res_name]
            idx_a = ref.index("N")
            idx_b = ref.index("CA")
            idx_c = ref.index("C")
            if max(idx_a, idx_b, idx_c) < len(present):
                return bool(present[idx_a] and present[idx_b]
                            and present[idx_c])
        except (ValueError, KeyError):
            pass
        idx_n = next((i for i, n in enumerate(names) if n == "N"), -1)
        idx_ca = next((i for i, n in enumerate(names) if n == "CA"), -1)
        idx_c = next((i for i, n in enumerate(names) if n == "C"), -1)
        if idx_n >= 0 and idx_ca >= 0 and idx_c >= 0:
            return bool(present[idx_n] and present[idx_ca] and present[idx_c])
        return False
    if mol_type in (chain_type_ids["DNA"],
                    chain_type_ids["RNA"]) and res_name in ref_atoms:
        try:
            ref = ref_atoms[res_name]
            idx_a = ref.index("C1'")
            idx_b = ref.index("C3'")
            idx_c = ref.index("C4'")
            if max(idx_a, idx_b, idx_c) < len(present):
                return bool(present[idx_a] and present[idx_b]
                            and present[idx_c])
        except (ValueError, KeyError):
            pass
        idx_c1 = next((i for i, n in enumerate(names) if n == "C1'"), -1)
        idx_c3 = next((i for i, n in enumerate(names) if n == "C3'"), -1)
        idx_c4 = next((i for i, n in enumerate(names) if n == "C4'"), -1)
        if idx_c1 >= 0 and idx_c3 >= 0 and idx_c4 >= 0:
            return bool(present[idx_c1] and present[idx_c3]
                        and present[idx_c4])
        return False
    return False


def _compute_collinear_mask(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """OSS featurizerv2 compute_collinear_mask: True where frame is not collinear."""
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1_n = v1 / (norm1 + 1e-6)
    v2_n = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1_n * v2_n, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return (mask_angle & mask_overlap1 & mask_overlap2).astype(np.float32)


def _convert_atom_name(name: str) -> tuple[int, int, int, int]:
    name = str(name).strip()
    name = [ord(c) - 32 for c in name][:4]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def process_token_features(
    tokens: list[Token],
    token_bonds: list[TokenBond],
    structure: Structure,
    override_method: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Build token-level feature tensors."""
    n = len(tokens)
    num_tok = max_tokens if max_tokens is not None else n
    pad_len = (num_tok - n) if max_tokens is not None and n < max_tokens else 0

    token_index = torch.arange(n, dtype=torch.long)
    residue_index = torch.tensor([t.res_idx for t in tokens], dtype=torch.long)
    asym_id = torch.tensor([t.asym_id for t in tokens], dtype=torch.long)
    entity_id = torch.tensor([t.entity_id for t in tokens], dtype=torch.long)
    sym_id = torch.tensor([t.sym_id for t in tokens], dtype=torch.long)
    mol_type = torch.tensor([t.mol_type for t in tokens], dtype=torch.long)
    res_type = torch.tensor([t.res_type for t in tokens], dtype=torch.long)
    res_type = one_hot(res_type, num_classes=num_tokens)
    disto_center = torch.tensor([t.disto_coords for t in tokens],
                                dtype=torch.float32)
    modified = torch.tensor([t.modified for t in tokens], dtype=torch.long)
    cyclic_period = torch.tensor([t.cyclic_period for t in tokens],
                                 dtype=torch.long)
    affinity_mask = torch.tensor([t.affinity_mask for t in tokens],
                                 dtype=torch.float32)

    method_key = ("x-ray diffraction"
                  if override_method is None else override_method.lower())
    method_id = method_types_ids.get(method_key, method_types_ids["other"])
    method_feature = torch.full((n, ), method_id, dtype=torch.long)

    pad_mask = torch.ones(n, dtype=torch.float32)
    resolved_mask = torch.tensor([t.resolved_mask for t in tokens],
                                 dtype=torch.float32)
    disto_mask = torch.tensor([t.disto_mask for t in tokens],
                              dtype=torch.float32)

    tok_to_idx = {t.token_idx: i for i, t in enumerate(tokens)}
    bonds = torch.zeros(num_tok, num_tok, dtype=torch.float32)
    bonds_type = torch.zeros(num_tok, num_tok, dtype=torch.long)
    for tb in token_bonds:
        i1 = tok_to_idx.get(tb.token_1)
        i2 = tok_to_idx.get(tb.token_2)
        if i1 is not None and i2 is not None:
            bonds[i1, i2] = 1
            bonds[i2, i1] = 1
            bonds_type[i1, i2] = tb.type
            bonds_type[i2, i1] = tb.type
    bonds = bonds.unsqueeze(-1)

    contact_conditioning_np = np.full((n, n),
                                      contact_conditioning_info["UNSELECTED"],
                                      dtype=np.int64)
    contact_threshold = np.zeros((n, n), dtype=np.float32)
    if np.all(contact_conditioning_np ==
              contact_conditioning_info["UNSELECTED"]):
        contact_conditioning_np = (contact_conditioning_np -
                                   contact_conditioning_info["UNSELECTED"] +
                                   contact_conditioning_info["UNSPECIFIED"])
    contact_conditioning = torch.from_numpy(contact_conditioning_np).long()
    contact_conditioning = one_hot(contact_conditioning,
                                   num_classes=len(contact_conditioning_info))
    contact_threshold = torch.from_numpy(contact_threshold).float()

    if pad_len > 0:
        token_index = pad_dim(token_index.unsqueeze(0), 1, pad_len).squeeze(0)
        residue_index = pad_dim(residue_index.unsqueeze(0), 1,
                                pad_len).squeeze(0)
        asym_id = pad_dim(asym_id.unsqueeze(0), 1, pad_len).squeeze(0)
        entity_id = pad_dim(entity_id.unsqueeze(0), 1, pad_len).squeeze(0)
        sym_id = pad_dim(sym_id.unsqueeze(0), 1, pad_len).squeeze(0)
        mol_type = pad_dim(mol_type.unsqueeze(0), 1, pad_len).squeeze(0)
        res_type = pad_dim(res_type.unsqueeze(0), 1, pad_len).squeeze(0)
        disto_center = pad_dim(disto_center.unsqueeze(0), 1,
                               pad_len).squeeze(0)
        modified = pad_dim(modified.unsqueeze(0), 1, pad_len).squeeze(0)
        cyclic_period = pad_dim(cyclic_period.unsqueeze(0), 1,
                                pad_len).squeeze(0)
        affinity_mask = pad_dim(affinity_mask.unsqueeze(0), 1,
                                pad_len).squeeze(0)
        method_feature = pad_dim(method_feature.unsqueeze(0), 1,
                                 pad_len).squeeze(0)
        pad_mask = pad_dim(pad_mask.unsqueeze(0), 1, pad_len).squeeze(0)
        resolved_mask = pad_dim(resolved_mask.unsqueeze(0), 1,
                                pad_len).squeeze(0)
        disto_mask = pad_dim(disto_mask.unsqueeze(0), 1, pad_len).squeeze(0)
        contact_conditioning = pad_dim(contact_conditioning, 0, pad_len)
        contact_conditioning = pad_dim(contact_conditioning, 1, pad_len)
        contact_threshold = pad_dim(contact_threshold, 0, pad_len)
        contact_threshold = pad_dim(contact_threshold, 1, pad_len)

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
        "type_bonds": bonds_type,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        "method_feature": method_feature,
        "modified": modified,
        "cyclic_period": cyclic_period,
        "affinity_token_mask": affinity_mask,
    }


def process_ensemble_features() -> dict[str, torch.Tensor]:
    """Single conformer: ensemble_ref_idxs = [0]."""
    return {"ensemble_ref_idxs": torch.tensor([0], dtype=torch.long)}


def process_atom_features(
    structure: Structure,
    tokens: list[Token],
    molecules: dict[str, Any],
    ensemble_ref_idxs: torch.Tensor,
    num_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    atoms_per_window_queries: int = 32,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Build atom-level feature tensors from structure and tokens."""
    from .const import chirality_type_ids, unk_chirality_type

    unk_chirality = chirality_type_ids[unk_chirality_type]
    n_tokens = len(tokens)
    # OSS-aligned: use structure.coords + ensemble offsets (ensemble_ref_idxs)
    ensemble_atom_starts = [
        int(structure.ensemble[int(e)]["atom_coord_idx"])
        for e in ensemble_ref_idxs
    ]
    num_ens = len(ensemble_atom_starts)
    atom_to_token = []
    token_to_rep_atom = []
    token_to_center_atom = []
    r_set_to_rep_atom = []
    ref_space_uid = []
    atom_name_list = []
    atom_element_list = []
    atom_charge_list = []
    atom_conformer_list = []
    atom_chirality_list = []
    atom_bfactor_list = [
    ]  # OSS-aligned: atom_data["bfactor"] built in token iteration order
    atom_plddt_list = [
    ]  # OSS-aligned: atom_data["plddt"] built in token iteration order
    backbone_feat_index = []
    frame_data = []
    resolved_frame_data = []
    coord_data_list = []
    disto_coords_list = []
    atom_idx = 0
    chain_res_ids = {}
    res_index_to_conf_id = {}

    for token_id, token in enumerate(tokens):
        chain_idx, res_id = token.asym_id, token.res_idx
        key = (chain_idx, res_id)
        if key not in chain_res_ids:
            chain_res_ids[key] = len(chain_res_ids)
        new_idx = chain_res_ids[key]
        mol = molecules.get(token.res_name)
        if mol is None:
            raise ValueError(f"Missing molecule for residue: {token.res_name}")
        atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}
        conf_ids = [int(c.GetId()) for c in mol.GetConformers()]
        if (chain_idx, res_id) not in res_index_to_conf_id:
            res_index_to_conf_id[(chain_idx, res_id)] = int(
                random.choice(conf_ids)) if conf_ids else 0
        conf_id = res_index_to_conf_id[(chain_idx, res_id)]
        conformer = mol.GetConformer(conf_id)

        start = token.atom_idx
        end = token.atom_idx + token.atom_num
        token_atoms = structure.atoms[start:end]

        ref_space_uid.extend([new_idx] * token.atom_num)
        atom_to_token.extend([token_id] * token.atom_num)

        # OSS-aligned: bfactor/plddt from structure in same order as atom_data (featurizerv2 line 1384 + 1419)
        atom_bfactor_list.extend(structure.bfactor[start:end].tolist())
        atom_plddt_list.extend(structure.plddt[start:end].tolist())

        for a in token_atoms:
            atom_name_list.append(_convert_atom_name(a.name))
            ref_atom = atom_name_to_ref.get(a.name)
            if ref_atom is not None:
                atom_element_list.append(ref_atom.GetAtomicNum())
                atom_charge_list.append(ref_atom.GetFormalCharge())
                pos = conformer.GetAtomPosition(ref_atom.GetIdx())
                atom_conformer_list.append((pos.x, pos.y, pos.z))
                atom_chirality_list.append(
                    chirality_type_ids.get(str(ref_atom.GetChiralTag()),
                                           unk_chirality))
            else:
                atom_element_list.append(a.element)
                atom_charge_list.append(a.charge)
                atom_conformer_list.append(a.conformer)
                atom_chirality_list.append(a.chirality)

        token_to_rep_atom.append(atom_idx + token.disto_idx - start)
        token_to_center_atom.append(atom_idx + token.center_idx - start)
        chain = structure.chains[token.asym_id]
        if chain.mol_type != chain_type_ids[
                "NONPOLYMER"] and token.resolved_mask:
            r_set_to_rep_atom.append(atom_idx + token.center_idx - start)

        if chain.mol_type == chain_type_ids["PROTEIN"]:
            for a in token_atoms:
                bi = (protein_backbone_atom_index.get(a.name, -1) +
                      1 if a.name in protein_backbone_atom_index else 0)
                backbone_feat_index.append(bi)
        elif chain.mol_type in (chain_type_ids["DNA"], chain_type_ids["RNA"]):
            for a in token_atoms:
                bi = (nucleic_backbone_atom_index.get(a.name, -1) + 1 +
                      len(protein_backbone_atom_index)
                      if a.name in nucleic_backbone_atom_index else 0)
                backbone_feat_index.append(bi)
        else:
            backbone_feat_index.extend([0] * token.atom_num)

        # OSS-aligned: coord_data and disto from structure.coords + ensemble offsets
        token_coords = np.array(
            [
                structure.coords[ea_start + start:ea_start + end]
                for ea_start in ensemble_atom_starts
            ],
            dtype=np.float32,
        )
        coord_data_list.append(token_coords)
        disto_coords_list.append(
            np.array(
                [
                    structure.coords[ea_start + token.disto_idx]
                    for ea_start in ensemble_atom_starts
                ],
                dtype=np.float32,
            ))
        if token.atom_num >= 3 and token.res_name in ref_atoms and ref_atoms[
                token.res_name][:3] == ["N", "CA", "C"]:
            frame_data.append([start, start + 1, start + 2])
        else:
            frame_data.append(
                [token.center_idx, token.center_idx, token.center_idx])
        # OSS-aligned: frame_resolved_mask from structure (featurizerv2 resolved_frame_data)
        resolved_frame_data.append(
            _frame_resolved_mask_oss(token_atoms, chain.mol_type,
                                     token.res_name, token.atom_num,
                                     token.res_type))
        atom_idx += token.atom_num

    # OSS compute_frames path: frame_resolved_mask = resolved_frame_data & mask_collinear (featurizerv2 line 1470, 176)
    coord_data = np.concatenate(coord_data_list, axis=1)
    frame_data_arr = np.array(frame_data, dtype=np.int64)
    frames_expanded = coord_data[0][frame_data_arr]  # (n_tokens, 3, 3)
    v1 = frames_expanded[:, 1] - frames_expanded[:, 0]
    v2 = frames_expanded[:, 1] - frames_expanded[:, 2]
    mask_collinear = _compute_collinear_mask(v1, v2)
    resolved_frame_data_np = np.array([float(x) for x in resolved_frame_data],
                                      dtype=np.float32)
    resolved_frame_data_np = resolved_frame_data_np * mask_collinear

    coord_data = np.concatenate(coord_data_list, axis=1)
    disto_coords_ensemble = np.stack(disto_coords_list, axis=0)
    disto_coords_ensemble = torch.from_numpy(disto_coords_ensemble).float()
    disto_coords_ensemble = disto_coords_ensemble.permute(1, 0, 2)

    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    t_center = torch.from_numpy(disto_coords_ensemble[0].numpy()).float()
    t_dists = torch.cdist(t_center, t_center)
    distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    disto_target = one_hot(distogram, num_classes=num_bins).unsqueeze(2)
    disto_target = disto_target.expand(-1, -1, num_ens, -1)

    atom_name_arr = np.array(atom_name_list, dtype=np.int32)
    atom_element_arr = np.array(atom_element_list, dtype=np.int64)
    atom_charge_arr = np.array(atom_charge_list, dtype=np.float32)
    atom_conformer_arr = np.array(atom_conformer_list, dtype=np.float32)
    atom_chirality_arr = np.array(atom_chirality_list, dtype=np.int64)
    backbone_feat_index = np.array(backbone_feat_index, dtype=np.int64)
    ref_space_uid = np.array(ref_space_uid, dtype=np.int64)
    # OSS-aligned: atom_data built in token iteration order (featurizerv2 atom_data.append(token_atoms); atom_data = np.concatenate(atom_data))
    # resolved_mask = atom_data["is_present"]; bfactor/plddt = atom_data["bfactor"], atom_data["plddt"]
    resolved_mask = torch.tensor(
        [structure.atoms[i].is_present for i in range(atom_idx)],
        dtype=torch.float32,
    )
    pad_mask = torch.ones(atom_idx, dtype=torch.float32)
    atom_to_token_t = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom_t = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom_t = torch.tensor(r_set_to_rep_atom, dtype=torch.long)
    token_to_center_atom_t = torch.tensor(token_to_center_atom,
                                          dtype=torch.long)
    bfactor = torch.tensor(atom_bfactor_list, dtype=torch.float32)
    plddt = torch.tensor(atom_plddt_list, dtype=torch.float32)

    ref_pos = torch.from_numpy(atom_conformer_arr).float()
    ref_atom_name_chars = torch.from_numpy(atom_name_arr).long()
    ref_element = torch.from_numpy(atom_element_arr).long()
    ref_charge = torch.from_numpy(atom_charge_arr).float()
    ref_chirality = torch.from_numpy(atom_chirality_arr).long()
    ref_space_uid_t = torch.from_numpy(ref_space_uid).long()
    coords = torch.from_numpy(coord_data).float()
    # OSS featurizerv2: apply center_random_augmentation per ref_space (deterministic under seed).
    ref_space_max = int(ref_space_uid_t.max().item())
    if ref_space_max >= 0:
        for i in range(ref_space_max + 1):
            included = ref_space_uid_t == i
            if included.sum() > 0 and resolved_mask[included].any():
                ref_pos[included] = _center_random_augmentation(
                    ref_pos[included].unsqueeze(0),
                    resolved_mask[included].unsqueeze(0),
                    s_trans=1.0,
                    augmentation=True,
                    centering=True,
                )[0]
    # Center ground-truth coords by resolved_mask (match OSS featurizerv2).
    # Note: coords values differ from OSS refs when refs are from YAML/sequence-only:
    # OSS schema builds StructureV2.coords from atoms["coords"], which is (0,0,0) in parse_polymer;
    # we build structure.coords from CCD conformer, so our coords are the ideal positions.
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]
    backbone_feat_index_t = torch.from_numpy(backbone_feat_index).long()

    backbone_feat_index_t = one_hot(
        backbone_feat_index_t,
        num_classes=1 + len(protein_backbone_atom_index) +
        len(nucleic_backbone_atom_index),
    )
    ref_atom_name_chars = one_hot(ref_atom_name_chars, num_classes=64)
    ref_element = one_hot(ref_element, num_classes=num_elements)
    atom_to_token_t = one_hot(atom_to_token_t, num_classes=n_tokens)
    token_to_rep_atom_t = one_hot(token_to_rep_atom_t, num_classes=atom_idx)
    r_set_to_rep_atom_t = one_hot(r_set_to_rep_atom_t, num_classes=atom_idx)
    token_to_center_atom_t = one_hot(token_to_center_atom_t,
                                     num_classes=atom_idx)

    pad_len_atom = (((atom_idx - 1) // atoms_per_window_queries + 1) *
                    atoms_per_window_queries - atom_idx)
    if max_atoms is not None:
        pad_len_atom = max_atoms - atom_idx
    if pad_len_atom > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len_atom)
        ref_pos = pad_dim(ref_pos, 0, pad_len_atom)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len_atom)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len_atom)
        ref_element = pad_dim(ref_element, 0, pad_len_atom)
        ref_charge = pad_dim(ref_charge, 0, pad_len_atom)
        ref_chirality = pad_dim(ref_chirality, 0, pad_len_atom)
        backbone_feat_index_t = pad_dim(backbone_feat_index_t, 0, pad_len_atom)
        ref_space_uid_t = pad_dim(ref_space_uid_t, 0, pad_len_atom)
        coords = pad_dim(coords, 1, pad_len_atom)
        atom_to_token_t = pad_dim(atom_to_token_t, 0, pad_len_atom)
        token_to_rep_atom_t = pad_dim(token_to_rep_atom_t, 1, pad_len_atom)
        token_to_center_atom_t = pad_dim(token_to_center_atom_t, 1,
                                         pad_len_atom)
        r_set_to_rep_atom_t = pad_dim(r_set_to_rep_atom_t, 1, pad_len_atom)
        bfactor = pad_dim(bfactor, 0, pad_len_atom)
        plddt = pad_dim(plddt, 0, pad_len_atom)
        atom_idx += pad_len_atom

    frames_idx = torch.tensor(frame_data, dtype=torch.long).unsqueeze(0)
    frame_resolved_mask = torch.from_numpy(
        resolved_frame_data_np).float().unsqueeze(0)
    if max_tokens is not None and n_tokens < max_tokens:
        pl = max_tokens - n_tokens
        disto_target = pad_dim(pad_dim(disto_target, 1, pl), 2, pl)
        disto_coords_ensemble = pad_dim(disto_coords_ensemble, 1, pl)
        frames_idx = pad_dim(frames_idx, 1, pl)
        frame_resolved_mask = pad_dim(frame_resolved_mask, 1, pl)

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_chirality": ref_chirality,
        "atom_backbone_feat": backbone_feat_index_t,
        "ref_space_uid": ref_space_uid_t,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token_t,
        "token_to_rep_atom": token_to_rep_atom_t,
        "r_set_to_rep_atom": r_set_to_rep_atom_t,
        "token_to_center_atom": token_to_center_atom_t,
        "disto_target": disto_target,
        "disto_coords_ensemble": disto_coords_ensemble,
        "bfactor": bfactor,
        "plddt": plddt,
        "frames_idx": frames_idx,
        "frame_resolved_mask": frame_resolved_mask,
    }


def _msa_from_parsed(
    msa_parsed: Optional[Any],
    num_residues: int,
    prot_letter_to_token: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (msa, deletion, paired) from MSAParsed.

    Paired is a binary flag: row 0 (query) = 1, rest = 0. Without taxonomy
    data there is no cross-chain pairing, matching OSS construct_paired_msa
    for monomer / no-taxonomy inputs.
    """
    seqs = msa_parsed.get("sequences") if msa_parsed is not None else None
    if seqs is None or (hasattr(seqs, '__len__') and len(seqs) == 0):
        msa = torch.zeros(1, num_residues, dtype=torch.long)
        deletion = torch.zeros(1, num_residues, dtype=torch.float32)
        paired = torch.ones(1, num_residues, dtype=torch.float32)
        return msa, deletion, paired
    raw_val = msa_parsed.get("raw")
    raw_list = raw_val if raw_val is not None else msa_parsed["sequences"]
    rows = []
    del_rows = []
    for raw in raw_list:
        res_types = []
        del_counts = []
        del_count = 0
        for c in raw:
            if c.islower():
                del_count += 1
            else:
                three = prot_letter_to_token.get(c.upper(), "UNK")
                res_types.append(token_ids.get(three, token_ids["UNK"]))
                del_counts.append(del_count)
                del_count = 0
        if len(res_types) == num_residues:
            rows.append(res_types)
            del_rows.append(del_counts)
    if not rows:
        msa = torch.zeros(1, num_residues, dtype=torch.long)
        deletion = torch.zeros(1, num_residues, dtype=torch.float32)
        paired = torch.ones(1, num_residues, dtype=torch.float32)
        return msa, deletion, paired
    msa = torch.tensor(rows, dtype=torch.long)
    deletion = torch.tensor(del_rows, dtype=torch.float32)
    n_rows = msa.shape[0]
    paired = torch.zeros(n_rows, num_residues, dtype=torch.float32)
    paired[0, :] = 1.0  # query row is always paired
    return msa, deletion, paired


def process_msa_features(
    tokens: list[Token],
    msa_parsed_per_chain: Optional[list[Optional[Any]]] = None,
    paired_msa_per_chain: Optional[list[Optional[Any]]] = None,
    max_seqs: int = 16384,
    max_paired: int = 8192,
    max_tokens: Optional[int] = None,
    pad_to_max_seqs: bool = False,
) -> dict[str, torch.Tensor]:
    """Build MSA feature tensors with optional taxonomy-paired rows."""
    from .const import prot_letter_to_token

    num_residues = len(tokens)
    if msa_parsed_per_chain is None or (hasattr(msa_parsed_per_chain,
                                                '__len__')
                                        and len(msa_parsed_per_chain) == 0):
        msa, deletion, paired = _msa_from_parsed(None, num_residues,
                                                 prot_letter_to_token)
    else:
        num_chains = max(t.asym_id for t in tokens) + 1
        residues_per_chain = [
            sum(1 for t in tokens if t.asym_id == i) for i in range(num_chains)
        ]

        # Parse unpaired MSA per chain.
        parts_msa, parts_del, parts_paired = [], [], []
        for i in range(num_chains):
            Lc = residues_per_chain[i]
            mp = msa_parsed_per_chain[i] if i < len(
                msa_parsed_per_chain) else None
            m, d, p = _msa_from_parsed(mp, Lc, prot_letter_to_token)
            parts_msa.append(m)
            parts_del.append(d)
            parts_paired.append(p)

        # Parse paired (taxonomy) MSA per chain.
        paired_parts_msa, paired_parts_del = [], []
        n_taxonomy_pairs = 0
        if paired_msa_per_chain is not None and len(paired_msa_per_chain) > 0:
            for i in range(num_chains):
                Lc = residues_per_chain[i]
                pp = (paired_msa_per_chain[i]
                      if i < len(paired_msa_per_chain) else None)
                pm, pd, _ = _msa_from_parsed(pp, Lc, prot_letter_to_token)
                paired_parts_msa.append(pm)
                paired_parts_del.append(pd)
            n_taxonomy_pairs = max(
                (pm.shape[0] - 1 for pm in paired_parts_msa), default=0)
            n_taxonomy_pairs = min(n_taxonomy_pairs, max_paired)

        # Heterodimer: multiple distinct MSA sources → taxonomy self-pair row.
        n_distinct = len(
            set(id(mp) for mp in msa_parsed_per_chain if mp is not None))
        is_heterodimer = num_chains > 1 and n_distinct > 1

        # Build combined per-chain MSA: [paired_non_query, unpaired_non_query].
        # OSS keeps paired seqs in the unpaired pool (visited set is a no-op),
        # so all non-query seqs appear in both paired rows AND unpaired rows.
        combined_msa, combined_del = [], []
        for i in range(num_chains):
            parts = [parts_msa[i][0:1]]  # query row
            parts_d = [parts_del[i][0:1]]
            if paired_parts_msa:
                pm = paired_parts_msa[i]
                pd = paired_parts_del[i]
                if pm.shape[0] > 1:
                    parts.append(pm[1:])
                    parts_d.append(pd[1:])
            if parts_msa[i].shape[0] > 1:
                parts.append(parts_msa[i][1:])
                parts_d.append(parts_del[i][1:])
            combined_msa.append(torch.cat(parts, dim=0))
            combined_del.append(torch.cat(parts_d, dim=0))

        max_non_query = max(cm.shape[0] - 1 for cm in combined_msa)
        # Row layout: query + self-pair? + taxonomy pairs + unpaired
        n_header = 1 + (1 if is_heterodimer else 0) + n_taxonomy_pairs
        max_unpaired = max(max_seqs - n_header, 0)
        max_non_query = min(max_non_query, max_unpaired)
        total_rows = n_header + max_non_query
        gap_id = token_ids["-"]
        msa = torch.full((total_rows, num_residues), gap_id, dtype=torch.long)
        deletion = torch.zeros(total_rows, num_residues, dtype=torch.float32)
        paired = torch.zeros(total_rows, num_residues, dtype=torch.float32)

        col = 0
        for i in range(num_chains):
            Lc = residues_per_chain[i]
            cm = combined_msa[i]
            cd = combined_del[i]
            # Row 0: query.
            msa[0, col:col + Lc] = cm[0]
            deletion[0, col:col + Lc] = cd[0]
            paired[0, col:col + Lc] = 1.0

            r_off = 1
            # Taxonomy self-pair (heterodimer): duplicate of query, paired=1.
            if is_heterodimer:
                msa[r_off, col:col + Lc] = cm[0]
                deletion[r_off, col:col + Lc] = cd[0]
                paired[r_off, col:col + Lc] = 1.0
                r_off += 1

            # Taxonomy-paired rows from paired MSA non-query seqs, paired=1.
            if paired_parts_msa:
                pm = paired_parts_msa[i]
                pd = paired_parts_del[i]
                n_avail = pm.shape[0] - 1  # non-query rows in this chain
                for j in range(n_taxonomy_pairs):
                    if j < n_avail:
                        msa[r_off + j, col:col + Lc] = pm[1 + j]
                        deletion[r_off + j, col:col + Lc] = pd[1 + j]
                    paired[r_off + j, col:col + Lc] = 1.0
            r_off += n_taxonomy_pairs

            # Unpaired rows: non-query sequences, capped to fit max_seqs.
            n_unpaired = min(cm.shape[0] - 1, max_non_query)
            if n_unpaired > 0:
                msa[r_off:r_off + n_unpaired,
                    col:col + Lc] = cm[1:1 + n_unpaired]
                deletion[r_off:r_off + n_unpaired,
                         col:col + Lc] = cd[1:1 + n_unpaired]
            col += Lc
    # Keep layout (N_MSA, L) to match Boltz2 featurizerv2; do not transpose.
    msa_one_hot = one_hot(msa, num_classes=num_tokens)
    msa_mask = torch.ones_like(msa, dtype=torch.float32)
    profile = msa_one_hot.float().mean(dim=0)
    has_deletion = deletion > 0
    deletion_val = np.pi / 2 * np.arctan(deletion.numpy() / 3)
    deletion_val = torch.from_numpy(deletion_val.astype(np.float32))
    deletion_mean = deletion_val.mean(dim=0)
    if pad_to_max_seqs and msa.shape[0] < max_seqs:
        pl = max_seqs - msa.shape[0]
        msa = pad_dim(msa, 0, pl, token_ids["-"])
        paired = pad_dim(paired, 0, pl)
        msa_mask = pad_dim(msa_mask, 0, pl)
        has_deletion = pad_dim(has_deletion.float(), 0, pl).bool()
        deletion_val = pad_dim(deletion_val, 0, pl)
        deletion_mean = pad_dim(deletion_mean.unsqueeze(0), 0, pl).squeeze(0)
    if max_tokens is not None and num_residues < max_tokens:
        pl = max_tokens - num_residues
        msa = pad_dim(msa, 1, pl, token_ids["-"])
        paired = pad_dim(paired, 1, pl)
        msa_mask = pad_dim(msa_mask, 1, pl)
        has_deletion = pad_dim(has_deletion.float(), 1, pl).bool()
        deletion_val = pad_dim(deletion_val, 1, pl)
        profile = pad_dim(profile, 0, pl)
        deletion_mean = pad_dim(deletion_mean, 0, pl)
    return {
        "msa": msa,
        "msa_paired": paired,
        "deletion_value": deletion_val,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "msa_mask": msa_mask,
    }


def load_dummy_templates_features(
        tdim: int, num_tokens_val: int) -> dict[str, torch.Tensor]:
    """Dummy template features (no templates)."""
    res_type = torch.zeros(tdim, num_tokens_val, dtype=torch.long)
    res_type = one_hot(res_type, num_classes=num_tokens)
    frame_rot = torch.zeros(tdim, num_tokens_val, 3, 3, dtype=torch.float32)
    frame_t = torch.zeros(tdim, num_tokens_val, 3, dtype=torch.float32)
    cb_coords = torch.zeros(tdim, num_tokens_val, 3, dtype=torch.float32)
    ca_coords = torch.zeros(tdim, num_tokens_val, 3, dtype=torch.float32)
    frame_mask = torch.zeros(tdim, num_tokens_val, dtype=torch.float32)
    cb_mask = torch.zeros(tdim, num_tokens_val, dtype=torch.float32)
    template_mask = torch.zeros(tdim, num_tokens_val, dtype=torch.float32)
    query_to_template = torch.zeros(tdim, num_tokens_val, dtype=torch.long)
    visibility_ids = torch.zeros(tdim, num_tokens_val, dtype=torch.float32)
    return {
        "template_restype": res_type,
        "template_frame_rot": frame_rot,
        "template_frame_t": frame_t,
        "template_cb": cb_coords,
        "template_ca": ca_coords,
        "template_mask_cb": cb_mask,
        "template_mask_frame": frame_mask,
        "template_mask": template_mask,
        "query_to_template": query_to_template,
        "visibility_ids": visibility_ids,
    }


def process_residue_constraint_features() -> dict[str, torch.Tensor]:
    """No constraints: empty tensors."""
    return {
        "rdkit_bounds_index": torch.empty((2, 0), dtype=torch.long),
        "rdkit_bounds_bond_mask": torch.empty(0, dtype=torch.bool),
        "rdkit_bounds_angle_mask": torch.empty(0, dtype=torch.bool),
        "rdkit_upper_bounds": torch.empty(0, dtype=torch.float32),
        "rdkit_lower_bounds": torch.empty(0, dtype=torch.float32),
        "chiral_atom_index": torch.empty((4, 0), dtype=torch.long),
        "chiral_reference_mask": torch.empty(0, dtype=torch.bool),
        "chiral_atom_orientations": torch.empty(0, dtype=torch.bool),
        "stereo_bond_index": torch.empty((4, 0), dtype=torch.long),
        "stereo_reference_mask": torch.empty(0, dtype=torch.bool),
        "stereo_bond_orientations": torch.empty(0, dtype=torch.bool),
        "planar_bond_index": torch.empty((6, 0), dtype=torch.long),
        "planar_ring_5_index": torch.empty((5, 0), dtype=torch.long),
        "planar_ring_6_index": torch.empty((6, 0), dtype=torch.long),
    }


def process_chain_feature_constraints(
    structure: Optional[Structure], ) -> dict[str, torch.Tensor]:
    """OSS-aligned: connected_chain/atom_index from bonds; symmetric_chain_index from entity_id (featurizerv2 lines 2018-2056)."""
    empty_2_0 = torch.empty((2, 0), dtype=torch.long)
    if structure is None:
        return {
            "connected_chain_index": empty_2_0,
            "connected_atom_index": empty_2_0,
            "symmetric_chain_index": empty_2_0,
        }
    if structure.bonds and len(structure.bonds) > 0:
        connected_chain_index, connected_atom_index = [], []
        for bond in structure.bonds:
            if bond.chain_1 == bond.chain_2:
                continue
            connected_chain_index.append([bond.chain_1, bond.chain_2])
            connected_atom_index.append([bond.atom_1, bond.atom_2])
        if connected_chain_index:
            connected_chain_index = torch.tensor(connected_chain_index,
                                                 dtype=torch.long).T
            connected_atom_index = torch.tensor(connected_atom_index,
                                                dtype=torch.long).T
        else:
            connected_chain_index = torch.empty((2, 0), dtype=torch.long)
            connected_atom_index = torch.empty((2, 0), dtype=torch.long)
    else:
        connected_chain_index = torch.empty((2, 0), dtype=torch.long)
        connected_atom_index = torch.empty((2, 0), dtype=torch.long)

    symmetric_chain_index_list: list[list[int]] = []
    chains = structure.chains
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            if chains[i].entity_id == chains[j].entity_id:
                symmetric_chain_index_list.append([i, j])
    if symmetric_chain_index_list:
        symmetric_chain_index = torch.tensor(symmetric_chain_index_list,
                                             dtype=torch.long).T
    else:
        symmetric_chain_index = torch.empty((2, 0), dtype=torch.long)
    return {
        "connected_chain_index": connected_chain_index,
        "connected_atom_index": connected_atom_index,
        "symmetric_chain_index": symmetric_chain_index,
    }


def process_contact_feature_constraints() -> dict[str, torch.Tensor]:
    return {
        "contact_pair_index": torch.empty((2, 0), dtype=torch.long),
        "contact_union_index": torch.empty(0, dtype=torch.long),
        "contact_negation_mask": torch.empty(0, dtype=torch.bool),
        "contact_thresholds": torch.empty(0, dtype=torch.float32),
    }
