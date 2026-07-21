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
"""OpenFold3 shared utilities: one-hot encoding, atom name encoding, etc."""

import math

import torch
import torch.nn.functional as F

from .const import (ATOM_NAME_TO_ELEMENT, ELEMENT_ATOMIC_NUMBER,
                    NUM_ATOM_NAME_CHARS, NUM_CHAR_CLASSES, NUM_ELEMENT_CLASSES)


def encode_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode integer indices.

    Args:
        x: [*] tensor of integer indices.
        num_classes: Number of classes.

    Returns:
        [*, num_classes] one-hot encoded tensor.
    """
    x_one_hot = torch.zeros(*x.shape,
                            num_classes,
                            device=x.device,
                            dtype=torch.int32)
    x_one_hot.scatter_(-1, x.unsqueeze(-1).long(), 1)
    return x_one_hot


def encode_atom_name_chars(atom_name: str) -> list[int]:
    """Encode an atom name as 4 integer character codes.

    Each character is encoded as ord(c) - 32. The name is right-padded with
    spaces to NUM_ATOM_NAME_CHARS (4) characters.

    Standard atom names (uppercase letters, digits, space) yield codes in
    [0, NUM_CHAR_CLASSES - 1] (i.e. [0, 63]) as consumed by
    ``encode_atom_name_chars_one_hot``. Characters outside that range would
    trigger an out-of-bounds error in the subsequent one-hot call.

    Args:
        atom_name: Atom name string (e.g. "CA", "NE2").

    Returns:
        List of NUM_ATOM_NAME_CHARS integer codes.
    """
    padded = atom_name.ljust(NUM_ATOM_NAME_CHARS)[:NUM_ATOM_NAME_CHARS]
    return [ord(c) - 32 for c in padded]


def encode_atom_name_chars_one_hot(atom_names: list[str]) -> torch.Tensor:
    """One-hot encode a list of atom names.

    Args:
        atom_names: List of atom name strings.

    Returns:
        [N_atoms, NUM_ATOM_NAME_CHARS, NUM_CHAR_CLASSES] int32 tensor.
    """
    codes = [encode_atom_name_chars(name) for name in atom_names]
    codes_tensor = torch.tensor(codes, dtype=torch.long)
    return F.one_hot(codes_tensor, NUM_CHAR_CLASSES).to(torch.int32)


def encode_element_one_hot(atom_names: list[str]) -> torch.Tensor:
    """One-hot encode element types from atom names.

    Uses atomic number - 1 as index (0-based). Unknown atom names fall back to
    Carbon via ATOM_NAME_TO_ELEMENT, and unknown elements fall back to Carbon
    (atomic number 6, index 5) via ELEMENT_ATOMIC_NUMBER.

    Args:
        atom_names: List of atom name strings.

    Returns:
        [N_atoms, NUM_ELEMENT_CLASSES] int32 tensor produced via F.one_hot on
        atomic_num - 1.
    """
    indices = []
    for name in atom_names:
        element = ATOM_NAME_TO_ELEMENT.get(name, "C")
        atomic_num = ELEMENT_ATOMIC_NUMBER.get(element, 6)  # default Carbon
        indices.append(atomic_num - 1)  # 0-indexed
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    return F.one_hot(idx_tensor, NUM_ELEMENT_CLASSES).to(torch.int32)


def deletion_matrix_from_raw(raw_sequences: list[str],
                             query_length: int) -> list[list[int]]:
    """Extract deletion counts from raw A3M sequences.

    In A3M format, lowercase letters represent insertions relative to the
    query. The deletion matrix counts consecutive lowercase characters before
    each aligned (uppercase or gap) position.

    Args:
        raw_sequences: List of raw A3M sequence strings.
        query_length: Length of the query sequence (number of aligned columns).

    Returns:
        List of lists, each of length query_length, with deletion counts.
    """
    deletion_matrix = []
    for raw_seq in raw_sequences:
        row = []
        deletion_count = 0
        for char in raw_seq:
            if char.islower():
                deletion_count += 1
            else:
                row.append(deletion_count)
                deletion_count = 0
        # Pad or truncate to query_length
        if len(row) < query_length:
            row.extend([0] * (query_length - len(row)))
        elif len(row) > query_length:
            row = row[:query_length]
        deletion_matrix.append(row)
    return deletion_matrix


def compute_deletion_value(deletion_matrix: torch.Tensor) -> torch.Tensor:
    """Compute scaled deletion value from raw deletion counts.

    Reproduces OSS's `core/data/pipelines/featurization/msa.py:106-108`:

        features["deletion_value"] = torch.atan(deletion_matrix / 3.0) * (
            2.0 / torch.acos(torch.zeros(1, device=deletion_matrix.device)) * 2
        ).to(torch.float32)

    Python operator precedence gives `(2.0 / acos(0)) * 2 = (2 / (π/2)) * 2 =
    (4/π) * 2 = 8/π ≈ 2.546`. The textbook formula would be `atan(x/3) * 2/π`
    (maps [0, ∞) → [0, 1)); OSS instead multiplies by `8/π`, mapping
    [0, ∞) → [0, 4). This is almost certainly a parenthesization bug in
    OSS, but the checkpoint was trained on these inflated values — see
    D-15 overturn for the profile-bug case (the same model-vs-OSS-feature
    parity argument applies to `deletion_value`).

    01-07 Cycle 7 Plan A': changed from `2/pi` (textbook) to `8/pi` (OSS
    reproduction) after `deletion_value` was promoted DETERMINISTIC and
    L1 comparison surfaced the ~4x scale mismatch.

    Args:
        deletion_matrix: [N_rows, N_tokens] int tensor of deletion counts.

    Returns:
        [N_rows, N_tokens] float32 tensor in [0, 4).
    """
    return (torch.atan(deletion_matrix.float() / 3.0) * (8.0 / math.pi)).to(
        torch.float32)


def _sample_rotations(shape: tuple, dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
    """Sample uniform random rotations via quaternion → rotation matrix.

    Matches AF3 Algorithm 19 / OSS sample_rotations().
    """
    quats = torch.randn(*shape, 4, dtype=dtype, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    # Quaternion to rotation matrix
    w, x, y, z = quats.unbind(-1)
    rots = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ],
                       dim=-1).reshape(*shape, 3, 3)
    return rots


def centre_random_augmentation(pos: torch.Tensor,
                               mask: torch.Tensor,
                               scale_trans: float = 1.0) -> torch.Tensor:
    """Centre, randomly rotate and translate conformer coordinates.

    Matches OSS centre_random_augmentation() (AF3 Algorithm 19).

    Args:
        pos: [*, N_atoms, 3] atom positions.
        mask: [*, N_atoms] validity mask (1=valid, 0=padding).
        scale_trans: Translation scaling factor.

    Returns:
        [*, N_atoms, 3] augmented positions.
    """
    rots = _sample_rotations(shape=pos.shape[:-2],
                             dtype=pos.dtype,
                             device=pos.device)
    trans = scale_trans * torch.randn(
        (*pos.shape[:-2], 3), dtype=pos.dtype, device=pos.device)

    mean_pos = torch.sum(
        pos * mask[..., None],
        dim=-2,
        keepdim=True,
    ) / torch.sum(mask[..., None], dim=-2, keepdim=True).clamp(min=1)

    pos_centered = pos - mean_pos
    pos_out = pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]
    pos_out = pos_out * mask[..., None]
    return pos_out


# ---------------------------------------------------------------------------
# Template feature math (direct-CIF path). Reimplements the OSS OpenFold-3
# featurization primitives (restype / distogram / unit-vector) exactly: given
# the same precursor arrays, outputs match ``featurize_template_structures_of3``
# up to float precision.
# ---------------------------------------------------------------------------


def create_template_restype(
    res_names,
    template_pseudo_beta_mask: torch.Tensor,
    resname_to_idx: dict,
    unk_idx: int,
    num_classes: int,
) -> torch.Tensor:
    """One-hot residue types for template tokens ([n_templ, n_tokens, C] int32).

    3-letter names -> indices (default UNK), one-hot over the 32-class vocab.
    ``res_names`` defaults to ``"GAP"`` for unaligned tokens. OSS does not gate
    restype on ``template_pseudo_beta_mask`` (kept in the signature for parity).
    """
    import numpy as np

    flat = np.asarray(res_names).reshape(-1)
    idx = np.fromiter((resname_to_idx.get(str(n), unk_idx) for n in flat),
                      dtype=np.int64,
                      count=flat.size).reshape(np.asarray(res_names).shape)
    restype_index = torch.tensor(idx, dtype=torch.int64)
    one_hot = torch.zeros(*restype_index.shape,
                          num_classes,
                          dtype=torch.int32)
    one_hot.scatter_(-1, restype_index.unsqueeze(-1), 1)
    return one_hot.to(torch.int32)


def create_template_distogram(
    pseudo_beta_atom_coords,
    pseudo_beta_mask: torch.Tensor,
    multichain_pair_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    n_bins: int,
    inf_value: float,
) -> torch.Tensor:
    """Template distogram [n_templ, n_tokens, n_tokens, n_bins] (float32).

    Squared pairwise pseudo-beta distances binned into ``n_bins`` squared-edge
    bins, masked by the pseudo-beta outer product and the chain pair mask.
    """
    import numpy as np

    coords = np.asarray(pseudo_beta_atom_coords)
    distogram = np.sum(
        (coords[..., None, :] - coords[..., None, :, :])**2,
        axis=-1,
        keepdims=True,
    )
    lower = np.linspace(min_bin, max_bin, n_bins)**2
    upper = np.concatenate([lower[1:],
                            np.array([inf_value], dtype=lower.dtype)],
                           axis=-1)
    binned = ((distogram > lower) * (distogram < upper)).astype(distogram.dtype)
    template_distogram = torch.tensor(binned, dtype=torch.float32)

    pb = pseudo_beta_mask
    pair = (pb[..., None] * pb[..., None, :])[..., None]
    return template_distogram * pair * multichain_pair_mask


def _rot3_from_two_vectors(e0: torch.Tensor,
                           e1: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt rotation from two vectors (OSS ``Rot3Array.from_two_vectors``).

    x-axis is ``e0`` normalized; the ``e1`` component orthogonal to x forms the
    y-axis; z = x cross y. Returns a [..., 3, 3] rotation whose columns are
    (x, y, z) axes — i.e. ``R @ v`` maps a local vector into the global frame.
    """
    eps = 1e-12
    x = e0 / e0.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (e1 * x).sum(dim=-1, keepdim=True)
    y = e1 - dot * x
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def create_template_unit_vector(
    frame_atom_coords,
    backbone_frame_mask: torch.Tensor,
    multichain_pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Template unit-vector feature [n_templ, n_tokens, n_tokens, 3] (float32).

    Builds a backbone rigid frame per token from N/CA/C, then expresses the
    direction to every other token's CA in the source token's local frame as a
    unit vector. Masked (NaN) frames contribute zeros.
    """
    import numpy as np

    coords = torch.nan_to_num(torch.tensor(np.asarray(frame_atom_coords),
                                           dtype=torch.float32),
                              nan=0.0)
    n_xyz = coords[:, :, 0, :]
    ca_xyz = coords[:, :, 1, :]
    c_xyz = coords[:, :, 2, :]

    # Rigid frame per token: rotation from (C-CA, N-CA), translation = CA.
    rot = _rot3_from_two_vectors(c_xyz - ca_xyz, n_xyz - ca_xyz)  # [T, N, 3, 3]
    trans = ca_xyz  # [T, N, 3]

    # Vector from source token i's frame origin to target token j's CA, rotated
    # into i's local frame: R_i^T @ (CA_j - CA_i).
    diff = trans[:, None, :, :] - trans[:, :, None, :]  # [T, N_i, N_j, 3]
    rot_t = rot.transpose(-1, -2)  # inverse rotation, [T, N, 3, 3]
    local = torch.einsum("tnij,tnmj->tnmi", rot_t, diff)  # [T, N_i, N_j, 3]

    norm = local.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit_vector = local / norm

    bb = backbone_frame_mask
    pair = (bb[..., None] * bb[..., None, :])[..., None]
    return unit_vector * pair * multichain_pair_mask
