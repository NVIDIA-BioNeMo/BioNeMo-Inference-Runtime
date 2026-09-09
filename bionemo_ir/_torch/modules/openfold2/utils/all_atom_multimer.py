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

# Copyright 2021 DeepMind Technologies Limited
# Modified by NVIDIA Corporation and affiliates.
"""Ops for all atom representations."""

from functools import partial

import torch

import bionemo_ir.pipeline.models.openfold2.const as rc
from bionemo_ir._torch.modules.openfold2.utils import geometry
from bionemo_ir._torch.utils import tensor as tensor_utils


def get_chi_atom_indices(device: torch.device):
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
        A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
        in the order specified in rc.restypes + unknown residue type
        at the end. For chi angles which are not defined on the residue, the
        positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    return torch.tensor(chi_atom_indices, device=device)


def compute_chi_angles(positions: geometry.Vec3Array, mask: torch.Tensor, aatype: torch.Tensor):
    """Computes the chi angles given all atom positions and the amino acid type.

    Args:
        positions: A Vec3Array of shape
            [num_res, rc.atom_type_num], with positions of
            atoms needed to calculate chi angles. Supports up to 1 batch dimension.
        mask: An optional tensor of shape
            [num_res, rc.atom_type_num] that masks which atom
            positions are set for each residue. If given, then the chi mask will be
            set to 1 for a chi angle only if the amino acid has that chi angle and all
            the chi atoms needed to calculate that chi angle are set. If not given
            (set to None), the chi mask will be set to 1 for a chi angle if the amino
            acid has that chi angle and whether the actual atoms needed to calculate
            it were set will be ignored.
        aatype: A tensor of shape [num_res] with amino acid type integer
            code (0 to 21). Supports up to 1 batch dimension.

    Returns:
        A tuple of tensors (chi_angles, mask), where both have shape
        [num_res, 4]. The mask masks out unused chi angles for amino acid
        types that have less than 4 chi angles. If atom_positions_mask is set, the
        chi mask will also mask out uncomputable chi angles.
    """

    # Don't assert on the num_res and batch dimensions as they might be unknown.
    assert positions.shape[-1] == rc.atom_type_num
    assert mask.shape[-1] == rc.atom_type_num
    no_batch_dims = len(aatype.shape) - 1

    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices(aatype.device)

    # DISCREPANCY: DeepMind doesn't remove the gaps here. I don't know why
    # theirs works.
    aatype_gapless = torch.clamp(aatype, max=20)

    # Select atoms to compute chis. Shape: [*, num_res, chis=4, atoms=4].
    atom_indices = chi_atom_indices[aatype_gapless]
    # Gather atom positions. Shape: [num_res, chis=4, atoms=4, xyz=3].
    chi_angle_atoms = positions.map_tensor_fn(
        partial(tensor_utils.batched_gather, inds=atom_indices, dim=-1, no_batch_dims=no_batch_dims + 1)
    )

    a, b, c, d = [chi_angle_atoms[..., i] for i in range(4)]

    chi_angles = geometry.dihedral_angle(a, b, c, d)

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(chi_angles_mask, device=aatype.device)
    # Compute the chi angle mask. Shape [num_res, chis=4].
    chi_mask = chi_angles_mask[aatype_gapless]

    # The chi_mask is set to 1 only when all necessary chi angle atoms were set.
    # Gather the chi angle atoms mask. Shape: [num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = tensor_utils.batched_gather(mask, atom_indices, dim=-1, no_batch_dims=no_batch_dims + 1)
    # Check if all 4 chi angle atoms were set. Shape: [num_res, chis=4].
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1)
    chi_mask = chi_mask * chi_angle_atoms_mask.to(chi_angles.dtype)

    return chi_angles, chi_mask


def make_transform_from_reference(
    a_xyz: geometry.Vec3Array, b_xyz: geometry.Vec3Array, c_xyz: geometry.Vec3Array
) -> geometry.Rigid3Array:
    """Returns rotation and translation matrices to convert from reference.

    Note that this method does not take care of symmetries. If you provide the
    coordinates in the non-standard way, the A atom will end up in the negative
    y-axis rather than in the positive y-axis. You need to take care of such
    cases in your code.

    Args:
        a_xyz: A Vec3Array.
        b_xyz: A Vec3Array.
        c_xyz: A Vec3Array.

    Returns:
        A Rigid3Array which, when applied to coordinates in a canonicalized
        reference frame, will give coordinates approximately equal
        the original coordinates (in the global frame).
    """
    rotation = geometry.Rot3Array.from_two_vectors(c_xyz - b_xyz, a_xyz - b_xyz)
    return geometry.Rigid3Array(rotation, b_xyz)


def make_backbone_affine(
    positions: geometry.Vec3Array,
    mask: torch.Tensor,
    aatype: torch.Tensor,
) -> tuple[geometry.Rigid3Array, torch.Tensor]:
    a = rc.atom_order["N"]
    b = rc.atom_order["CA"]
    c = rc.atom_order["C"]

    rigid_mask = mask[..., a] * mask[..., b] * mask[..., c]

    rigid = make_transform_from_reference(
        a_xyz=positions[..., a],
        b_xyz=positions[..., b],
        c_xyz=positions[..., c],
    )

    return rigid, rigid_mask
