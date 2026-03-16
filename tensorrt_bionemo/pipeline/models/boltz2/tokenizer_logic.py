# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenize Structure into Tokens and TokenBonds (no OSS imports)."""

from __future__ import annotations

import numpy as np

from .const import (Structure, Token, TokenBond, chain_type_ids, token_ids,
                    unk_token)


def compute_frame(
    n: tuple[float, float, float],
    ca: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    """Compute backbone frame from N, CA, C. Returns (3x3 rotation as 3 tuples, translation)."""
    n_arr = np.array(n, dtype=np.float64)
    ca_arr = np.array(ca, dtype=np.float64)
    c_arr = np.array(c, dtype=np.float64)
    v1 = c_arr - ca_arr
    v2 = n_arr - ca_arr
    e1 = v1 / (np.linalg.norm(v1) + 1e-10)
    u2 = v2 - e1 * np.dot(e1, v2)
    e2 = u2 / (np.linalg.norm(u2) + 1e-10)
    e3 = np.cross(e1, e2)
    rot = np.column_stack([e1, e2, e3])
    t = ca_arr
    rot_tuples = (
        (float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2])),
        (float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2])),
        (float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])),
    )
    t_tup = (float(t[0]), float(t[1]), float(t[2]))
    return rot_tuples, t_tup


def tokenize_structure(
        struct: Structure) -> tuple[list[Token], list[TokenBond]]:
    """
    Tokenize a Structure into list of Token and list of TokenBond.
    Supports protein chains only (standard residues = one token per residue).
    """
    tokens: list[Token] = []
    token_bonds: list[TokenBond] = []
    atom_to_token: dict[int, int] = {}
    token_idx = 0
    coords = struct.coords
    # Offset for coords: we use direct indexing (single conformer)
    offset = 0

    chains = [c for c, m in zip(struct.chains, struct.mask) if m]
    for chain in chains:
        res_start = chain.res_idx
        res_end = chain.res_idx + chain.res_num
        is_protein = chain.mol_type == chain_type_ids["PROTEIN"]
        affinity_mask = False

        for res in struct.residues[res_start:res_end]:
            atom_start = res.atom_idx
            atom_end = res.atom_idx + res.atom_num

            if res.is_standard:
                center_coords = (
                    float(coords[offset + res.atom_center, 0]),
                    float(coords[offset + res.atom_center, 1]),
                    float(coords[offset + res.atom_center, 2]),
                )
                disto_coords = (
                    float(coords[offset + res.atom_disto, 0]),
                    float(coords[offset + res.atom_disto, 1]),
                    float(coords[offset + res.atom_disto, 2]),
                )
                is_present = res.is_present
                is_disto_present = res.is_present

                frame_rot: tuple[tuple[float, float, float], ...] = (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
                frame_t = (0.0, 0.0, 0.0)
                frame_mask = False

                if is_protein and res.atom_num >= 3:
                    # N, CA, C are first three in ref_atoms
                    a0 = struct.atoms[res.atom_idx]
                    a1 = struct.atoms[res.atom_idx + 1]
                    a2 = struct.atoms[res.atom_idx + 2]
                    frame_mask = bool(a1.is_present and a2.is_present
                                      and a0.is_present)
                    if frame_mask:
                        frame_rot, frame_t = compute_frame(
                            a0.conformer, a1.conformer, a2.conformer)
                        frame_t = (float(frame_t[0]), float(frame_t[1]),
                                   float(frame_t[2]))

                token = Token(
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
                    affinity_mask=affinity_mask,
                )
                tokens.append(token)
                for i in range(atom_start, atom_end):
                    atom_to_token[i] = token_idx
                token_idx += 1
            else:
                # Non-standard (e.g. modified): use unk token, one token per residue
                unk_id = token_ids[unk_token["PROTEIN"]]
                center_coords = (
                    float(coords[offset + res.atom_center, 0]),
                    float(coords[offset + res.atom_center, 1]),
                    float(coords[offset + res.atom_center, 2]),
                )
                token = Token(
                    token_idx=token_idx,
                    atom_idx=res.atom_idx,
                    atom_num=res.atom_num,
                    res_idx=res.res_idx,
                    res_type=unk_id,
                    res_name=res.name,
                    sym_id=chain.sym_id,
                    asym_id=chain.asym_id,
                    entity_id=chain.entity_id,
                    mol_type=chain.mol_type,
                    center_idx=res.atom_center,
                    disto_idx=res.atom_disto,
                    center_coords=center_coords,
                    disto_coords=center_coords,
                    resolved_mask=res.is_present,
                    disto_mask=res.is_present,
                    modified=True,
                    frame_rot=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0,
                                                                  1.0)),
                    frame_t=(0.0, 0.0, 0.0),
                    frame_mask=False,
                    cyclic_period=chain.cyclic_period,
                    affinity_mask=affinity_mask,
                )
                tokens.append(token)
                for i in range(atom_start, atom_end):
                    atom_to_token[i] = token_idx
                token_idx += 1

    for bond in struct.bonds:
        if bond.atom_1 not in atom_to_token or bond.atom_2 not in atom_to_token:
            continue
        t1 = atom_to_token[bond.atom_1]
        t2 = atom_to_token[bond.atom_2]
        token_bonds.append(
            TokenBond(token_1=t1, token_2=t2, type=bond.type + 1))

    return tokens, token_bonds
