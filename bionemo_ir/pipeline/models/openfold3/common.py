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
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from bionemo_ir._torch.layers.random_augmentation import _quaternion_components_to_matrix
from bionemo_ir.pipeline.utils._rng import _torch_generator
from bionemo_ir.pipeline.utils.atom import encode_atom_name_chars as _encode_atom_name_chars
from bionemo_ir.pipeline.utils.atom import encode_atom_name_chars_one_hot as _encode_atom_name_chars_one_hot


def encode_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode integer indices.

    Args:
        x: [*] tensor of integer indices.
        num_classes: Number of classes.

    Returns:
        [*, num_classes] one-hot encoded tensor.
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device, dtype=torch.int32)
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
    return _encode_atom_name_chars(atom_name)


def encode_atom_name_chars_one_hot(atom_names: list[str]) -> torch.Tensor:
    """One-hot encode a list of atom names.

    Args:
        atom_names: List of atom name strings.

    Returns:
        [N_atoms, NUM_ATOM_NAME_CHARS, NUM_CHAR_CLASSES] int32 tensor.
    """
    return _encode_atom_name_chars_one_hot(atom_names)


def compute_deletion_value(deletion_matrix: torch.Tensor) -> torch.Tensor:
    """Compute scaled deletion value from raw deletion counts.

    Reproduces upstream `featurize_msa_of3`
    (`core/data/pipelines/featurization/msa.py`):

        features["deletion_value"] = torch.atan(deletion_matrix / 3.0) * (
            2.0 / torch.acos(torch.zeros(1, device=deletion_matrix.device)) * 2
        ).to(torch.float32)

    Python operator precedence gives `(2.0 / acos(0)) * 2 = (2 / (π/2)) * 2 =
    (4/π) * 2 = 8/π ≈ 2.546`. The textbook formula would be `atan(x/3) * 2/π`
    (maps [0, ∞) → [0, 1)); upstream instead multiplies by `8/π`, mapping
    [0, ∞) → [0, 4). This looks like a parenthesization slip upstream, but
    the released checkpoint was trained on these inflated values, so we
    reproduce `8/π` deliberately. The same parity argument applies to the
    MSA profile in `feature_generators.py` — do not "correct" either one to
    the textbook formula without retraining.

    Args:
        deletion_matrix: [N_rows, N_tokens] int tensor of deletion counts.

    Returns:
        [N_rows, N_tokens] float32 tensor in [0, 4).
    """
    return (torch.atan(deletion_matrix.float() / 3.0) * (8.0 / math.pi)).to(torch.float32)


def centre_random_augmentation_blocks(pos: torch.Tensor, block_sizes: Sequence[int]) -> torch.Tensor:
    """Centre, randomly rotate and translate consecutive atom blocks in one pass.

    Matches OSS centre_random_augmentation() (AF3 Algorithm 19) applied to
    each block in order, up to float32 rounding: each block draws the same rotation and
    translation from the request-local RNG, but the centering and rotation
    reductions run batched.

    Args:
        pos: [N_atoms, 3] CPU float32 positions; block ``b`` owns the next
            ``block_sizes[b]`` rows.
        block_sizes: Positive atom count of each block, summing to ``N_atoms``.

    Returns:
        [N_atoms, 3] augmented positions.
    """
    if not block_sizes:
        return pos.clone()
    # Per block, randn(7) yields the values of randn((1, 4)) then randn(3); one
    # randn((n_blocks, 7)) would take a different vectorized sampling path.
    noise = torch.stack([torch.randn(7, dtype=pos.dtype, generator=_torch_generator()) for _ in block_sizes])
    quaternions = noise[:, :4] / noise[:, :4].norm(dim=-1, keepdim=True)
    rotations = _quaternion_components_to_matrix(*quaternions.unbind(-1), 2.0)
    translations = noise[:, 4:]

    sizes = torch.tensor(block_sizes)
    block_of_atom = torch.repeat_interleave(torch.arange(len(block_sizes)), sizes)
    centres = torch.zeros(len(block_sizes), 3, dtype=pos.dtype).index_add_(0, block_of_atom, pos)
    centres = centres / sizes.to(pos.dtype)[:, None]
    centred = pos - centres[block_of_atom]
    return torch.einsum("ai,aji->aj", centred, rotations[block_of_atom]) + translations[block_of_atom]


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
    idx = np.fromiter((resname_to_idx.get(str(n), unk_idx) for n in flat), dtype=np.int64, count=flat.size).reshape(
        np.asarray(res_names).shape
    )
    restype_index = torch.tensor(idx, dtype=torch.int64)
    one_hot = torch.zeros(*restype_index.shape, num_classes, dtype=torch.int32)
    one_hot.scatter_(-1, restype_index.unsqueeze(-1), 1)
    return one_hot.to(torch.int32)


_DISTOGRAM_MIN_TOKENS = 512


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
    if (
        coords.ndim == 3
        and coords.shape[-1] == 3
        and coords.shape[1] >= _DISTOGRAM_MIN_TOKENS
        and coords.dtype in (np.float32, np.float64)
        and pseudo_beta_mask.shape == coords.shape[:2]
        and multichain_pair_mask.shape == (1, coords.shape[1], coords.shape[1], 1)
        and n_bins > 0
        and pseudo_beta_mask.device.type == multichain_pair_mask.device.type == "cpu"
        and pseudo_beta_mask.dtype == multichain_pair_mask.dtype == torch.float32
        and not pseudo_beta_mask.requires_grad
        and not multichain_pair_mask.requires_grad
        and torch.all((pseudo_beta_mask == 0) | (pseudo_beta_mask == 1))
        and torch.all((multichain_pair_mask == 0) | (multichain_pair_mask == 1))
        and not torch.any(torch.signbit(pseudo_beta_mask))
        and not torch.any(torch.signbit(multichain_pair_mask))
    ):
        lower = np.linspace(min_bin, max_bin, n_bins) ** 2
        if np.all(np.isfinite(lower)) and np.all(lower[1:] > lower[:-1]):
            return _distogram_rows(coords, pseudo_beta_mask, multichain_pair_mask, lower, inf_value)
    if coords.dtype in (np.float32, np.float64):
        # Same left-to-right sum as ``np.sum`` over the last axis, without the
        # [..., N, N, 3] temporaries. Other dtypes may accumulate wider.
        squares = [(coords[..., :, None, k] - coords[..., None, :, k]) ** 2 for k in range(3)]
        distances = squares[0] + squares[1] + squares[2]
        del squares
    else:
        distances = np.sum((coords[..., None, :] - coords[..., None, :, :]) ** 2, axis=-1)
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2
    upper = np.concatenate([lower[1:], np.array([inf_value], dtype=lower.dtype)], axis=-1)
    pb = pseudo_beta_mask
    pair = (pb[..., None] * pb[..., None, :])[..., None]
    if n_bins and np.all(np.isfinite(lower)) and np.all(lower[1:] > lower[:-1]):
        indices = np.searchsorted(lower, distances, side="left") - 1
        valid = (indices >= 0) & (distances < upper[np.clip(indices, 0, n_bins - 1)])
        binned = np.zeros((*distances.shape, n_bins), dtype=np.float32)
        rows = np.flatnonzero(valid.ravel())
        if (
            pair.device.type == multichain_pair_mask.device.type == "cpu"
            and pair.dtype == multichain_pair_mask.dtype == torch.float32
            and torch.broadcast_shapes(pair.shape, multichain_pair_mask.shape) == (*distances.shape, 1)
            and not pair.requires_grad
            and not multichain_pair_mask.requires_grad
            and not torch.any(torch.signbit(pair))
            and not torch.any(torch.signbit(multichain_pair_mask))
            and torch.all((pair == 0) | (pair == 1))
            and torch.all((multichain_pair_mask == 0) | (multichain_pair_mask == 1))
        ):
            # Mask sparse entries before dense materialization.
            weights = (pair * multichain_pair_mask).numpy().reshape(-1)
            binned.reshape(-1, n_bins)[rows, indices.ravel()[rows]] = weights[rows]
            return torch.as_tensor(binned)
        binned.reshape(-1, n_bins)[rows, indices.ravel()[rows]] = 1.0
    else:
        # Nonmonotone edges can describe overlapping bins.
        distogram = distances[..., None]
        binned = ((distogram > lower) * (distogram < upper)).astype(np.float32)
    template_distogram = torch.as_tensor(binned)

    shape = torch.broadcast_shapes(template_distogram.shape, pair.shape, multichain_pair_mask.shape)
    same_layout = shape == template_distogram.shape and template_distogram.numel()
    if same_layout and pair.dtype == multichain_pair_mask.dtype == torch.float32:
        # Same float32 products in place; skips two dense temporaries.
        return template_distogram.mul_(pair).mul_(multichain_pair_mask)
    return template_distogram * pair * multichain_pair_mask


_DISTOGRAM_LOCK = Lock()
_DISTOGRAM_POOL: tuple[int, ThreadPoolExecutor] | None = None


def _run_distogram_batch(fill_rows: Callable[[tuple[int, int]], None], bounds: list[tuple[int, int]]) -> None:
    global _DISTOGRAM_POOL
    with _DISTOGRAM_LOCK:
        pid = os.getpid()
        if _DISTOGRAM_POOL is None or _DISTOGRAM_POOL[0] != pid:
            cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
            _DISTOGRAM_POOL = pid, ThreadPoolExecutor(max_workers=min(8, cores), thread_name_prefix="bioir-distogram")
        list(_DISTOGRAM_POOL[1].map(fill_rows, bounds))


def _reset_distogram_pool() -> None:
    """Reset inherited state only in the forked child.

    Parent locks and in-flight work remain unchanged. External C forks must
    invoke Python's at-fork hooks; other libraries retain their fork limits.
    """
    global _DISTOGRAM_LOCK, _DISTOGRAM_POOL
    _DISTOGRAM_LOCK = Lock()
    _DISTOGRAM_POOL = None


def _lock_distogram_pool() -> None:
    _DISTOGRAM_LOCK.acquire()


def _unlock_distogram_pool() -> None:
    _DISTOGRAM_LOCK.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_lock_distogram_pool,
        after_in_parent=_unlock_distogram_pool,
        after_in_child=_reset_distogram_pool,
    )


def _distogram_rows(
    coords: "np.ndarray", pb: torch.Tensor, mc: torch.Tensor, lower: "np.ndarray", inf_value: float
) -> torch.Tensor:
    import numpy as np

    n_templ, n_tokens, _ = coords.shape
    n_bins = lower.size
    upper = np.concatenate([lower[1:], np.array([inf_value], dtype=lower.dtype)])
    masks, pair_mask = pb.numpy(), mc.numpy()[0, ..., 0]
    output = np.zeros((n_templ, n_tokens, n_tokens, n_bins), dtype=np.float32)
    active = np.flatnonzero(masks.any(axis=1))
    if not active.size:
        return torch.from_numpy(output)
    coords, masks = coords[active], masks[active]

    def fill_rows(bounds: tuple[int, int]) -> None:
        start, end = bounds
        allowed = (masks[:, start:end, None] != 0) & (masks[:, None, :] != 0)
        allowed &= pair_mask[start:end] != 0
        ti, ii, ji = np.nonzero(allowed)
        squares = [(coords[ti, ii + start, k] - coords[ti, ji, k]) ** 2 for k in range(3)]
        distances = squares[0] + squares[1] + squares[2]
        del squares
        indices = np.searchsorted(lower, distances, side="left") - 1
        valid = (indices >= 0) & (distances < upper[np.clip(indices, 0, n_bins - 1)])
        output[active[ti[valid]], ii[valid] + start, ji[valid], indices[valid]] = 1.0

    workers = min(8, max(1, n_tokens // 64))
    edges = np.linspace(0, n_tokens, workers + 1, dtype=int)
    bounds = [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:], strict=True)]
    _run_distogram_batch(fill_rows, bounds)
    return torch.from_numpy(output)


def _rot3_from_two_vectors(e0: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
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

    coords = torch.nan_to_num(torch.tensor(np.asarray(frame_atom_coords), dtype=torch.float32), nan=0.0)
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
    if (
        pair.dtype == multichain_pair_mask.dtype == torch.float32
        and torch.broadcast_shapes(unit_vector.shape, pair.shape, multichain_pair_mask.shape) == unit_vector.shape
    ):
        return unit_vector.mul_(pair).mul_(multichain_pair_mask)
    return unit_vector * pair * multichain_pair_mask
