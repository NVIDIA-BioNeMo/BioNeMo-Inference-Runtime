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

"""Noise-aligned chirality for RFProteína's single representation.

A stereocentre reaches the velocity network as packed unit-scale vectors, three per slot, written in
the frame of the supplied conformer. A checkpoint trained with ``nn.noise_align_chirality`` was
instead shown them in the frame of the *current noisy state*: each frame's vectors are rotated to
best match the directions its neighbours point in now, and optionally moved to the frame's noisy
position. Reading the supplied vectors unchanged feeds the trunk a geometry the rollout has not
reached, and every step differs from what the checkpoint saw in training. The alignment has no
parameters, so a checkpoint's weights cannot say whether it was trained with it; the host that holds
the training configuration decides.
"""

from __future__ import annotations

import torch

#: Packed stereochemical vectors, three per slot: ``[B, N, 3 * S]``.
CHIRALITY = "chirality"
#: Atoms that define each slot's frame: two origin tokens, then up to ``K`` neighbour tokens, ``-1``
#: where absent. ``[B, N, S, 2 + K]``.
CHIRAL_FRAME_ATOMS = "chiral_frame_atoms"
#: Owner of each slot, ``anchor_token * S + source_slot``, or ``-1`` for a slot nothing transforms.
#: A slot copied from another keeps its source's owner. ``[B, N, S]``.
CHIRAL_SLOT_OWNER = "chiral_slot_owner"

_ORIGIN_ATOMS = 2
_MIN_NORM = 1e-8


def proper_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """Fit proper rotations from source-transpose-times-target covariances.

    Args:
        covariance: ``[..., 3, 3]`` matrices ``source^T @ target``.

    Returns:
        ``[..., 3, 3]`` rotations applied to row vectors as ``source @ rotation``. A fit that is not
        unique, such as one from two neighbours, gets the optimum the SVD selects, which is what the
        checkpoint was trained on.
    """
    with torch.autocast(device_type=covariance.device.type, enabled=False):
        left, singular, right_t = torch.linalg.svd(covariance.float())
        signs = torch.ones_like(singular)
        signs[..., -1] = torch.linalg.det(left @ right_t).sign()
        return (left * signs[..., None, :]) @ right_t


@torch.no_grad()
def align_chirality(
    vectors: torch.Tensor,
    anchors: torch.Tensor,
    padding_mask: torch.Tensor,
    frame_atoms: torch.Tensor,
    slot_owner: torch.Tensor,
    *,
    rotate: bool = True,
    translate: bool = False,
) -> torch.Tensor:
    """Express packed chirality vectors in the frame of the current noisy state.

    Args:
        vectors: ``[B, N, 3 * S]`` supplied vectors, unit scale.
        anchors: ``[B, N, 3]`` noisy backbone anchors in nanometres, the ``bb_ca`` state.
        padding_mask: ``[B, N]`` mask of real tokens.
        frame_atoms: ``[B, N, S, 2 + K]`` frame definitions, see :data:`CHIRAL_FRAME_ATOMS`.
        slot_owner: ``[B, N, S]`` slot owners, see :data:`CHIRAL_SLOT_OWNER`.
        rotate: Fit each frame's orientation to its noisy neighbour directions.
        translate: Add the noisy position of the frame's origin atoms, which is their midpoint.

    Returns:
        ``[B, N, 3 * S]`` float32 vectors, zero on padding. Slots without an owner keep the supplied
        vectors. With both options off the input is returned as it came.

    Raises:
        ValueError: If a frame names a padding token, or a supplied vector or noisy anchor it uses is
            not finite.
    """
    if not (rotate or translate):
        return vectors
    batch, tokens, width = vectors.shape
    slots = width // 3
    valid = padding_mask.to(torch.bool)
    frames = (frame_atoms[..., 0] >= 0) & valid[..., None]
    batch_index, token_index, slot_index = torch.where(frames)  # each [F]
    if batch_index.numel() == 0:
        return vectors.float() * valid[..., None]

    # A host's bf16 autocast must not reach the fit: a rotation from rounded products is not a rotation.
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        anchors = anchors.float()
        origins = frame_atoms[batch_index, token_index, slot_index, :_ORIGIN_ATOMS]  # [F, 2]
        neighbors = frame_atoms[batch_index, token_index, slot_index, _ORIGIN_ATOMS:]  # [F, K]
        present = neighbors >= 0
        neighbors = neighbors.clamp_min(0)  # a missing neighbour reads token 0 and is masked below
        packed = vectors.float().reshape(batch, tokens, slots, 3)
        supplied = packed[batch_index[:, None], neighbors, slot_index[:, None]]  # [F, K, 3]
        origin = anchors[batch_index[:, None], origins].mean(-2, keepdim=True)  # [F, 1, 3]
        noisy = anchors[batch_index[:, None], neighbors] - origin  # [F, K, 3]

        names_real_tokens = (valid[batch_index[:, None], neighbors] | ~present).all() & valid[
            batch_index[:, None], origins
        ].all()
        finite = (torch.isfinite(supplied).all(-1) | ~present).all() & (torch.isfinite(noisy).all(-1) | ~present).all()
        # One host read for both checks; the cause is only worked out when something is wrong.
        if not bool(torch.stack([names_real_tokens, finite]).all()):
            if not bool(names_real_tokens):
                raise ValueError("chiral frame references padding tokens")
            raise ValueError("nonfinite chiral conditioning or noisy coordinates")

        # Frame ids 1..F in the order found above; 0 stands for "no transform".
        frame_id = torch.zeros(batch, tokens * slots, dtype=torch.long, device=vectors.device)
        frame_id[batch_index, token_index * slots + slot_index] = torch.arange(
            1, batch_index.numel() + 1, device=vectors.device
        )
        owners = slot_owner.long()
        slot_frame = frame_id[
            torch.arange(batch, device=vectors.device)[:, None, None], owners.clamp_min(0)
        ]  # [B, N, S]

        aligned = packed
        if rotate:
            supplied = torch.where(present[..., None], supplied, 0.0)
            noisy = torch.where(present[..., None], noisy, 0.0)
            noisy = noisy / torch.linalg.vector_norm(noisy, dim=-1, keepdim=True).clamp_min(_MIN_NORM)
            rotation = proper_rotation(supplied.transpose(-1, -2) @ noisy)  # [F, 3, 3]
            rotations = torch.cat([torch.eye(3, device=vectors.device)[None], rotation])  # [F + 1, 3, 3]
            aligned = torch.einsum("bnsx,bnsxy->bnsy", packed, rotations[slot_frame])
        if translate:
            translations = torch.cat([origin.new_zeros(1, 3), origin[:, 0]])  # [F + 1, 3]
            aligned = aligned + translations[slot_frame]
        aligned = torch.where((owners >= 0)[..., None], aligned, packed)
        return aligned.reshape(batch, tokens, width) * valid[..., None]
