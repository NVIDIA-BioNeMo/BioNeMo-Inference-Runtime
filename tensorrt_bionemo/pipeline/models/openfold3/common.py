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
