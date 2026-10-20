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
"""Tensor-only RFProteína partial-autoencoder encoder.

This ports the released legacy encoder feature contract without depending on
Atomworks, Lightning, or the OSS package. It maps an existing atom37 protein
structure to the clean local-latent endpoint used by partial redesign.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.attention_backend.interface import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import precompute_single_masks
from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    build_bias_mega_offset,
    build_bias_mega_weight,
    precompute_pair_biases,
)
from bionemo_ir._torch.layers.transformers.local_latents_decoder import ATOM37_SIZE, _LinearFactory
from bionemo_ir._torch.layers.transformers.local_latents_transformer import RFProteinaTrunkLayer
from bionemo_ir._torch.layers.transition import Transition
from bionemo_ir._torch.sampling.random import randn
from bionemo_ir.configs import BaseConfig
from bionemo_ir.runtime.buffers import PreallocatedBuffers
from bionemo_ir.utils import str_dtype_to_torch

# Protein residue order is the legacy OpenFold/RFProteína order:
# A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, unknown.
_CHI_ATOM_INDICES = (
    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 11), (3, 5, 11, 23), (5, 11, 23, 32)),
    ((0, 1, 3, 5), (1, 3, 5, 16), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 16), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 10), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 11), (3, 5, 11, 26), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 11), (3, 5, 11, 26), (0, 0, 0, 0)),
    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 14), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 6), (1, 3, 6, 12), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 12), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 11), (3, 5, 11, 19), (5, 11, 19, 35)),
    ((0, 1, 3, 5), (1, 3, 5, 18), (3, 5, 18, 19), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 12), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 11), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 8), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 9), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 12), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 5), (1, 3, 5, 12), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 1, 3, 6), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
)
_CHI_ANGLES_MASK = (
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 0, 0),
    (1, 1, 1, 0),
    (1, 1, 1, 0),
    (0, 0, 0, 0),
    (1, 1, 0, 0),
    (1, 1, 0, 0),
    (1, 1, 0, 0),
    (1, 1, 1, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 0, 0),
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 0, 0),
    (0, 0, 0, 0),
)


def _normalize_last_dim(value: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(value, dim=-1, keepdim=True)
    return value / torch.clamp(norm, min=1e-8)


def _signed_dihedral_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b0 = b - a
    b1 = c - b
    b2 = d - c
    n1 = _normalize_last_dim(torch.linalg.cross(b0, b1, dim=-1))
    n2 = _normalize_last_dim(torch.linalg.cross(b1, b2, dim=-1))
    cos_angle = torch.linalg.vecdot(n1, n2, dim=-1)
    n1_cross_n2 = torch.linalg.cross(n1, n2, dim=-1)
    sin_magnitude = torch.linalg.norm(n1_cross_n2, dim=-1)
    sign = torch.sign(torch.linalg.vecdot(n1_cross_n2, b1, dim=-1))
    return torch.atan2(sign * sin_magnitude, cos_angle)


def _bond_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b0 = _normalize_last_dim(b - a)
    b1 = _normalize_last_dim(c - a)
    cos_angle = torch.linalg.vecdot(b0, b1, dim=-1)
    sin_angle = torch.linalg.norm(torch.linalg.cross(b0, b1, dim=-1), dim=-1)
    return torch.atan2(sin_angle, cos_angle)


def _bin_indices(value: torch.Tensor, lower: float, upper: float, num_limits: int) -> torch.Tensor:
    limits = torch.linspace(lower, upper, num_limits, device=value.device, dtype=value.dtype)
    return torch.bucketize(value, limits)


def _backbone_torsion_features(
    coordinates: torch.Tensor, encoding_mask: torch.Tensor, token_index: torch.Tensor
) -> torch.Tensor:
    batch, num_tokens = encoding_mask.shape
    n_atom = coordinates[:, :, 0]
    ca_atom = coordinates[:, :, 1]
    c_atom = coordinates[:, :, 2]
    psi = _signed_dihedral_angle(n_atom[:, :-1], ca_atom[:, :-1], c_atom[:, :-1], n_atom[:, 1:])
    omega = _signed_dihedral_angle(ca_atom[:, :-1], c_atom[:, :-1], n_atom[:, 1:], ca_atom[:, 1:])
    phi = _signed_dihedral_angle(c_atom[:, :-1], n_atom[:, 1:], ca_atom[:, 1:], c_atom[:, 1:])
    angles = torch.stack((psi, omega, phi), dim=-1)
    sequential = token_index[:, 1:] - token_index[:, :-1] == 1
    pair_valid = encoding_mask[:, :-1] & encoding_mask[:, 1:] & sequential
    angles = torch.where(pair_valid.unsqueeze(-1), angles, torch.zeros_like(angles))
    angles = torch.cat((angles, torch.zeros(batch, 1, 3, device=coordinates.device, dtype=coordinates.dtype)), dim=1)
    bins = _bin_indices(angles, -torch.pi, torch.pi, 20)
    return F.one_hot(bins, num_classes=21).flatten(start_dim=-2).to(coordinates.dtype)


def _torsion_sin_cos(points: torch.Tensor) -> torch.Tensor:
    """OpenFold frame convention for ``[..., 4, 3]`` torsion atoms."""
    origin = points[..., 2, :]
    e1 = F.normalize(points[..., 1, :] - origin, dim=-1, eps=1e-8)
    v2 = points[..., 0, :] - origin
    v2 = v2 - torch.sum(e1 * v2, dim=-1, keepdim=True) * e1
    e2 = F.normalize(v2, dim=-1, eps=1e-8)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    relative = points[..., 3, :] - origin
    sin_cos = torch.stack((torch.sum(e3 * relative, dim=-1), torch.sum(e2 * relative, dim=-1)), dim=-1)
    return sin_cos / torch.sqrt(torch.sum(torch.square(sin_cos), dim=-1, keepdim=True) + 1e-8)


def _sidechain_angle_features(
    coordinates: torch.Tensor, atom_mask: torch.Tensor, residue_type: torch.Tensor
) -> torch.Tensor:
    output_dtype = coordinates.dtype
    coords = coordinates.double()
    coordinate_mask = atom_mask.double()
    residue_type = residue_type.clamp(min=0, max=20).long()
    indices = torch.as_tensor(_CHI_ATOM_INDICES, device=coordinates.device)[residue_type]
    gather_index = indices.unsqueeze(-1).expand(*indices.shape, 3)
    expanded_coords = coords.unsqueeze(-3).expand(-1, -1, 4, -1, -1)
    chi_coords = torch.gather(expanded_coords, -2, gather_index)
    expanded_mask = coordinate_mask.unsqueeze(-2).expand(-1, -1, 4, -1)
    chi_atom_mask = torch.gather(expanded_mask, -1, indices)
    chi_exists = torch.as_tensor(_CHI_ANGLES_MASK, device=coordinates.device, dtype=coordinate_mask.dtype)[residue_type]
    chi_mask = chi_exists * torch.prod(chi_atom_mask, dim=-1)
    sin_cos = _torsion_sin_cos(chi_coords) * chi_mask.unsqueeze(-1)
    angles = torch.atan2(sin_cos[..., 0], sin_cos[..., 1]) * chi_mask
    bins = _bin_indices(angles, -torch.pi, torch.pi, 20)
    angle_features = F.one_hot(bins, num_classes=21).to(coords.dtype) * chi_mask.unsqueeze(-1)
    return torch.cat((angle_features.flatten(start_dim=-2), chi_mask), dim=-1).to(output_dtype)


def _single_features(
    coordinates: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_type: torch.Tensor,
    encoding_mask: torch.Tensor,
    token_index: torch.Tensor,
    chain_breaks: torch.Tensor | None,
    num_residue_types: int,
) -> torch.Tensor:
    dtype = coordinates.dtype
    if chain_breaks is None:
        chain_breaks = torch.zeros_like(encoding_mask, dtype=dtype)
    else:
        chain_breaks = chain_breaks.to(device=coordinates.device, dtype=dtype)

    residue = residue_type.clamp(min=0, max=num_residue_types - 1)
    residue_feature = F.one_hot(residue.long(), num_classes=num_residue_types).to(dtype)
    residue_feature *= encoding_mask.unsqueeze(-1)

    coordinate_mask = atom_mask.to(dtype)
    masked_coordinates = coordinates * coordinate_mask.unsqueeze(-1)
    absolute = torch.cat((masked_coordinates.flatten(start_dim=-2), coordinate_mask), dim=-1)
    ca_coordinates = masked_coordinates[:, :, 1]
    relative_coordinates = (masked_coordinates - ca_coordinates.unsqueeze(-2)) * coordinate_mask.unsqueeze(-1)
    relative = torch.cat((relative_coordinates.flatten(start_dim=-2), coordinate_mask), dim=-1)
    backbone = _backbone_torsion_features(coordinates, encoding_mask, token_index)
    sidechain = _sidechain_angle_features(coordinates, atom_mask, residue_type)
    features = torch.cat((chain_breaks.unsqueeze(-1), residue_feature, absolute, relative, backbone, sidechain), dim=-1)
    return features * encoding_mask.unsqueeze(-1)


def _pair_feature_indices(
    coordinates: torch.Tensor,
    atom_mask: torch.Tensor,
    token_index: torch.Tensor,
    sequence_separation_bins: int,
    distance_bins: int,
    orientation_bins: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    separation = token_index.unsqueeze(-1) - token_index.unsqueeze(-2)
    separation_limit = sequence_separation_bins / 2.0 - 1
    separation_indices = _bin_indices(
        separation.to(torch.float32), -separation_limit, separation_limit, sequence_separation_bins - 1
    )

    n_atom = coordinates[:, :, 0]
    ca_atom = coordinates[:, :, 1]
    c_atom = coordinates[:, :, 2]
    cb_atom = coordinates[:, :, 3]
    ca_i = ca_atom.unsqueeze(2)
    atoms_j = (n_atom.unsqueeze(1), ca_atom.unsqueeze(1), c_atom.unsqueeze(1), cb_atom.unsqueeze(1))
    distances = [torch.linalg.norm(ca_i - atom_j, dim=-1) for atom_j in atoms_j]
    distances[-1] = distances[-1] * atom_mask[:, None, :, 3].to(distances[-1].dtype)
    distance_indices = tuple(_bin_indices(distance, 0.1, 2.0, distance_bins - 1) for distance in distances)

    n_i, ca_i, cb_i = (value.unsqueeze(2) for value in (n_atom, ca_atom, cb_atom))
    n_j, ca_j, cb_j = (value.unsqueeze(1) for value in (n_atom, ca_atom, cb_atom))
    angles = (
        _signed_dihedral_angle(n_i, ca_i, cb_i, cb_j),
        _signed_dihedral_angle(n_j, ca_j, cb_j, cb_i),
        _bond_angle(ca_i, cb_i, cb_j),
        _bond_angle(ca_j, cb_j, cb_i),
        _signed_dihedral_angle(ca_i, cb_i, cb_j, ca_j),
    )
    orientation_indices = tuple(_bin_indices(angle, -torch.pi, torch.pi, orientation_bins - 1) for angle in angles)
    return separation_indices, distance_indices, orientation_indices


def _left_align_transformer_tracks(
    single: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Move valid tokens to a prefix while preserving their relative order.

    CuTeDSL pairwise attention consumes only the number of valid keys and
    therefore requires ``1...1 0...0`` masks. Encoder masks can contain holes
    after intersecting padding with ``can_be_encoded`` and protein token types.
    The transformer is permutation equivariant when both pair axes move with
    the single track, so align those tracks and restore output order later.
    """
    mask = mask.to(torch.bool)
    if mask.shape[-1] < 2 or bool(torch.all(mask[..., :-1] >= mask[..., 1:])):
        return single, pair, mask, None

    permutation = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)
    single = torch.gather(
        single,
        1,
        permutation.unsqueeze(-1).expand(-1, -1, single.shape[-1]),
    )
    batch_index = torch.arange(pair.shape[0], device=pair.device)[:, None, None]
    pair = pair[batch_index, permutation[:, :, None], permutation[:, None, :]]
    mask = torch.gather(mask, 1, permutation)
    return single, pair, mask, permutation


def _restore_token_order(value: torch.Tensor, permutation: torch.Tensor | None) -> torch.Tensor:
    """Scatter a left-aligned transformer output back to input token order."""
    if permutation is None:
        return value
    index = permutation
    for _ in range(value.ndim - 2):
        index = index.unsqueeze(-1)
    return torch.empty_like(value).scatter_(1, index.expand_as(value), value)


class RFProteinaLatentEncoder(nn.Module):
    """Encode atom37 protein structures into RFProteína local latents."""

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.transformer_dtype = str_dtype_to_torch(config.transformer_dtype)
        self.num_residue_types = config.num_residue_types
        self.latent_dim = config.latent_dim

        self.init_repr_factory = _LinearFactory(config.dense_input_dim, config.token_dim, self.dtype)
        self.token_type_embedding = (
            nn.Embedding(config.num_token_kinds, config.token_type_embedding_dim, dtype=self.dtype)
            if config.input_schema == "unified"
            else None
        )
        # Not the released embeddings: each row is an embedding already multiplied by one atom37 slot's
        # columns of the single projection, so a lookup lands in ``token_dim``. Conversion builds them
        # (``fold_atom_chemistry``) and the projection keeps only the dense columns.
        chemistry = config.chemistry_input_dim > 0
        self.element_embedding = (
            nn.EmbeddingBag(ATOM37_SIZE * config.num_elements, config.token_dim, mode="sum", dtype=self.dtype)
            if chemistry
            else None
        )
        self.atom_charge_embedding = (
            nn.EmbeddingBag(ATOM37_SIZE * config.num_atom_charges, config.token_dim, mode="sum", dtype=self.dtype)
            if chemistry
            else None
        )
        self.pair_rep_factory = (
            _LinearFactory(config.pair_input_dim, config.pair_dim, self.dtype) if config.use_pair_bias else None
        )
        condition_hidden = int(config.condition_dim * config.condition_transition_expansion)
        self.transition_c_1 = Transition(config.condition_dim, condition_hidden, dtype=self.dtype, normalize=False)
        self.transition_c_2 = Transition(config.condition_dim, condition_hidden, dtype=self.dtype, normalize=False)
        self.transformer_layers = nn.ModuleList(
            [
                RFProteinaTrunkLayer(
                    layer_idx=index,
                    token_dim=config.token_dim,
                    pair_dim=config.pair_dim,
                    condition_dim=config.condition_dim,
                    num_heads=config.num_heads,
                    transition_expansion=config.transition_expansion,
                    backend=config.pairwise_attention_backend,
                    mask_inf=config.mask_inf,
                    use_qk_norm=config.use_qk_norm,
                    eps=config.norm_epsilon,
                    dtype=self.transformer_dtype,
                    norm_type="layer_norm",
                    cond_norm_bias=True,
                    use_pair_bias=config.use_pair_bias,
                    mask_left_aligned=True,
                    sanitize_masked_inputs=False,
                    require_inference_mode=False,
                )
                for index in range(config.num_layers)
            ]
        )
        self.latent_decoder_mean_n_log_scale = nn.Sequential(
            nn.LayerNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
            nn.Linear(config.token_dim, 2 * config.latent_dim, bias=False, dtype=self.dtype),
        )
        self.precompute_bias = config.precompute_bias
        self._num_heads = config.num_heads
        self._norm_epsilon = config.norm_epsilon
        self._bias_pad_multiple = 8 if config.pairwise_attention_backend == "CuTeDSL" else -1

    def _encoding_mask(
        self,
        mask: torch.Tensor,
        can_be_encoded: torch.Tensor | None,
        token_type: torch.Tensor | None,
    ) -> torch.Tensor:
        # Narrow out of place: a bool mask is its own ``.to(torch.bool)``, so
        # ``&=`` here would write through to the caller's tensor.
        encoding_mask = mask.to(torch.bool)
        if can_be_encoded is not None:
            if can_be_encoded.shape != mask.shape:
                raise ValueError(
                    f"can_be_encoded must have shape {tuple(mask.shape)}, got {tuple(can_be_encoded.shape)}"
                )
            encoding_mask = encoding_mask & can_be_encoded.to(torch.bool)
        if token_type is not None and self.config.input_schema == "legacy":
            if token_type.shape != mask.shape:
                raise ValueError(f"token_type must have shape {tuple(mask.shape)}, got {tuple(token_type.shape)}")
            encoding_mask = encoding_mask & (token_type == 0)
        return encoding_mask

    def _project_pair_features(
        self,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        token_index: torch.Tensor,
        encoding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pair_rep_factory is None:
            raise RuntimeError("pair feature projection requires use_pair_bias=True")
        separation, distances, orientations = _pair_feature_indices(
            coordinates,
            atom_mask,
            token_index,
            self.config.sequence_separation_bins,
            self.config.distance_bins,
            self.config.orientation_bins,
        )
        weight = self.pair_rep_factory.linear_out.weight
        offset = 0

        def project(index: torch.Tensor, width: int, valid: torch.Tensor) -> torch.Tensor:
            nonlocal offset
            table = weight[:, offset : offset + width].transpose(0, 1)
            offset += width
            return F.embedding(index, table) * valid.unsqueeze(-1)

        pair_valid = encoding_mask.unsqueeze(-1) & encoding_mask.unsqueeze(-2)
        ca_valid = atom_mask[:, :, 1].to(torch.bool) & encoding_mask
        coordinate_pair_valid = ca_valid.unsqueeze(-1) & ca_valid.unsqueeze(-2)
        # Mirror the released feature's source-CB mask, including its
        # historical directional behavior.
        orientation_valid = coordinate_pair_valid & atom_mask[:, :, 3].to(torch.bool).unsqueeze(-1)
        pair = project(separation, self.config.sequence_separation_bins, pair_valid)
        for distance in distances:
            pair += project(distance, self.config.distance_bins, coordinate_pair_valid)
        for orientation in orientations:
            pair += project(orientation, self.config.orientation_bins, orientation_valid)
        if offset != self.config.pair_input_dim:
            raise RuntimeError(
                f"encoder pair features consumed {offset} channels, expected {self.config.pair_input_dim}"
            )
        return pair * pair_valid.unsqueeze(-1)

    def _atom_chemistry_features(
        self,
        atom_mask: torch.Tensor,
        element: torch.Tensor | None,
        charge: torch.Tensor | None,
        bond_counts: torch.Tensor | None,
        expected_tokens: tuple[int, int],
        device: torch.device,
        token_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-atom chemistry: incident bond counts as projected features, element and charge as lookups.

        Each is read at every atom37 slot, so a residue contributes one block per slot whether or not it
        fills it, and slots it does not occupy contribute zeros. The released projection reads element and
        charge as embeddings concatenated per slot. It is linear and bias-free, so slot ``s`` adds
        ``W_s @ embedding[k]`` for the element or charge ``k`` found there, a function of one small integer.
        Conversion has stored that product as one table row, so the sum over slots is a single bag lookup in
        the projected width and the ``37 * (element + charge)``-wide chemistry features are never built.

        Args:
            token_valid: ``[B, N]`` tokens the encoder keeps; the lookup is zero on the rest.

        Returns:
            Bond counts flattened ``[B, N, 37 * bins]`` for the projection, and the chemistry contribution
            to the single representation ``[B, N, token_dim]``.
        """
        slots = (*expected_tokens, ATOM37_SIZE)
        valid = atom_mask.to(device=device, dtype=torch.bool)

        def checked(value: torch.Tensor | None, name: str, trailing: tuple[int, ...] = ()) -> torch.Tensor:
            expected = slots + trailing
            if value is None:
                return torch.zeros(expected, device=device, dtype=torch.long if not trailing else self.dtype)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
            return value.to(device=device)

        counts = checked(bond_counts, "incident_bond_type_counts", (self.config.incident_bond_type_bins,))
        counts = counts.to(self.dtype) * valid.unsqueeze(-1)

        elements = checked(element, "atom37_element").long().masked_fill(~valid, 0)
        if torch.any((elements < 0) | (elements >= self.config.num_elements)):
            # A row index past the slot's block would read the next slot's table instead of failing.
            raise ValueError(f"atom37_element must lie in [0, {self.config.num_elements - 1}] on occupied slots")
        charges = checked(charge, "atom37_charge").long()
        charges = (charges + self.config.atom_charge_offset).clamp(0, self.config.num_atom_charges - 1)
        charges = charges.masked_fill(~valid, 0)

        weights = (valid & token_valid.to(device=device, dtype=torch.bool).unsqueeze(-1)).to(self.dtype)
        weights = weights.reshape(-1, ATOM37_SIZE)
        slot = torch.arange(ATOM37_SIZE, device=device)
        element_rows = (elements + slot * self.config.num_elements).reshape(-1, ATOM37_SIZE)
        charge_rows = (charges + slot * self.config.num_atom_charges).reshape(-1, ATOM37_SIZE)
        chemistry = self.element_embedding(element_rows, per_sample_weights=weights)
        chemistry += self.atom_charge_embedding(charge_rows, per_sample_weights=weights)
        return counts.flatten(start_dim=-2), chemistry.reshape(*expected_tokens, -1)

    def build_representations(
        self,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        can_be_encoded: torch.Tensor | None = None,
        token_type: torch.Tensor | None = None,
        token_index: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
        chain_breaks: torch.Tensor | None = None,
        chain_iid: torch.Tensor | None = None,
        atom37_element: torch.Tensor | None = None,
        atom37_charge: torch.Tensor | None = None,
        incident_bond_type_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Build projected single/pair tracks and the effective encoder mask."""
        if coordinates.ndim != 4 or coordinates.shape[-2:] != (37, 3):
            raise ValueError(f"coordinates must have shape [B, N, 37, 3], got {tuple(coordinates.shape)}")
        batch, num_tokens = coordinates.shape[:2]
        expected_tokens = (batch, num_tokens)
        for name, value in (("atom_mask", atom_mask), ("residue_type", residue_type), ("mask", mask)):
            expected = expected_tokens + ((37,) if name == "atom_mask" else ())
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

        encoding_mask = self._encoding_mask(mask, can_be_encoded, token_type)
        invalid_residue = encoding_mask & ((residue_type < 0) | (residue_type >= self.num_residue_types))
        if torch.any(invalid_residue):
            values = torch.unique(residue_type[invalid_residue]).detach().cpu().tolist()
            raise ValueError(
                f"encodable residue_type values must be in [0, {self.num_residue_types - 1}], got {values}"
            )
        if token_index is None:
            token_index = torch.arange(1, num_tokens + 1, device=coordinates.device, dtype=torch.long)
            token_index = token_index.unsqueeze(0).expand(batch, -1)
        elif token_index.shape != expected_tokens:
            raise ValueError(f"token_index must have shape {expected_tokens}, got {tuple(token_index.shape)}")
        if residue_index is None:
            # OSS backbone torsions fall back to dense row positions when
            # residue_pdb_idx is absent. Pair separation still uses the
            # independently supplied within-index-group token positions.
            residue_index = torch.arange(1, num_tokens + 1, device=coordinates.device, dtype=torch.long)
            residue_index = residue_index.unsqueeze(0).expand(batch, -1)
        elif residue_index.shape != expected_tokens:
            raise ValueError(f"residue_index must have shape {expected_tokens}, got {tuple(residue_index.shape)}")

        chemistry = None
        if self.config.input_schema == "unified":
            if token_type is None:
                token_type = torch.zeros(expected_tokens, device=coordinates.device, dtype=torch.long)
            elif token_type.shape != expected_tokens:
                raise ValueError(f"token_type must have shape {expected_tokens}, got {tuple(token_type.shape)}")
            token_type = token_type.to(device=coordinates.device, dtype=torch.long)
            if torch.any((token_type < 0) | (token_type >= self.token_type_embedding.num_embeddings)):
                raise ValueError("token_type contains an unsupported unified token kind")
            if chain_iid is None:
                chain_iid = torch.zeros(expected_tokens, device=coordinates.device, dtype=self.dtype)
            elif chain_iid.shape != expected_tokens:
                raise ValueError(f"chain_iid must have shape {expected_tokens}, got {tuple(chain_iid.shape)}")
            coordinate_mask = atom_mask.to(dtype=coordinates.dtype)
            masked_coordinates = coordinates * coordinate_mask.unsqueeze(-1)
            absolute = torch.cat((masked_coordinates.flatten(start_dim=-2), coordinate_mask), dim=-1)
            ca_coordinates = masked_coordinates[:, :, 1]
            relative_coordinates = (masked_coordinates - ca_coordinates.unsqueeze(-2)) * coordinate_mask.unsqueeze(-1)
            relative = torch.cat((relative_coordinates.flatten(start_dim=-2), coordinate_mask), dim=-1)
            parts = [
                self.token_type_embedding(token_type),
                absolute,
                relative,
                chain_iid.to(device=coordinates.device, dtype=self.dtype).unsqueeze(-1),
            ]
            if self.config.atom_chemistry:
                counts, chemistry = self._atom_chemistry_features(
                    atom_mask,
                    atom37_element,
                    atom37_charge,
                    incident_bond_type_counts,
                    expected_tokens,
                    coordinates.device,
                    encoding_mask,
                )
                parts.append(counts)
            features = torch.cat(parts, dim=-1)
        else:
            features = _single_features(
                coordinates,
                atom_mask,
                residue_type,
                encoding_mask,
                residue_index,
                chain_breaks,
                self.num_residue_types,
            )
        valid = encoding_mask.unsqueeze(-1).to(self.dtype)
        single = self.init_repr_factory(features.to(self.dtype), valid)
        if chemistry is not None:
            single = single + chemistry
        pair = (
            self._project_pair_features(coordinates, atom_mask, token_index, encoding_mask)
            if self.config.use_pair_bias
            else None
        )
        return single, pair, encoding_mask

    def _precompute_all_biases(self, pair: torch.Tensor) -> list[torch.Tensor]:
        return precompute_pair_biases(
            pair.to(self.transformer_dtype),
            build_bias_mega_weight(self.transformer_layers, attn_attr="mha"),
            len(self.transformer_layers),
            self._num_heads,
            self._norm_epsilon,
            self._bias_pad_multiple,
            bias_offset=build_bias_mega_offset(self.transformer_layers, attn_attr="mha"),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_type: torch.Tensor,
        mask: torch.Tensor,
        can_be_encoded: torch.Tensor | None = None,
        token_type: torch.Tensor | None = None,
        token_index: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
        chain_breaks: torch.Tensor | None = None,
        chain_iid: torch.Tensor | None = None,
        atom37_element: torch.Tensor | None = None,
        atom37_charge: torch.Tensor | None = None,
        incident_bond_type_counts: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sample: bool = True,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        sampling_stream: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode nanometre atom37 coordinates.

        ``z_latent`` is sampled from the encoder posterior by default. Pass a
        generator for reproducibility or ``sample=False`` to use its mean.
        ``sampling_stream`` draws that sample through
        :func:`bionemo_ir._torch.sampling.random.randn`, so a caller replaying a
        reference implementation's random sequence sees it; production leaves it
        off and draws with ``torch.randn`` directly.
        """
        single, pair, encoding_mask = self.build_representations(
            coordinates,
            atom_mask,
            residue_type,
            mask,
            can_be_encoded=can_be_encoded,
            token_type=token_type,
            token_index=token_index,
            residue_index=residue_index,
            chain_breaks=chain_breaks,
            chain_iid=chain_iid,
            atom37_element=atom37_element,
            atom37_charge=atom37_charge,
            incident_bond_type_counts=incident_bond_type_counts,
        )
        output_encoding_mask = encoding_mask
        permutation = None
        if self.config.pairwise_attention_backend == "CuTeDSL" and pair is not None:
            single, pair, encoding_mask, permutation = _left_align_transformer_tracks(single, pair, encoding_mask)
        precomputed_single_masks = precompute_single_masks(
            self.config.pairwise_attention_backend,
            encoding_mask,
            inf=self.config.mask_inf,
        )
        condition = torch.zeros(
            *encoding_mask.shape, self.config.condition_dim, device=coordinates.device, dtype=self.dtype
        )
        condition = self.transition_c_1(condition, encoding_mask)
        condition = self.transition_c_2(condition, encoding_mask)
        single = single.to(self.transformer_dtype)
        if pair is not None:
            pair = pair.to(self.transformer_dtype)
        condition = condition.to(self.transformer_dtype)

        if self.precompute_bias:
            if pair is None:
                raise RuntimeError("precompute_bias requires a pair representation")
            biases = self._precompute_all_biases(pair)
            for layer in self.transformer_layers:
                layer.mha.bias_proj = False
            try:
                for index, layer in enumerate(self.transformer_layers):
                    single = layer(
                        single,
                        biases[index],
                        condition,
                        encoding_mask,
                        metadata=metadata,
                        buffers=buffers,
                        precomputed_single_masks=precomputed_single_masks,
                    )
            finally:
                for layer in self.transformer_layers:
                    layer.mha.bias_proj = True
        else:
            for layer in self.transformer_layers:
                single = layer(
                    single,
                    pair,
                    condition,
                    encoding_mask,
                    metadata=metadata,
                    buffers=buffers,
                    precomputed_single_masks=precomputed_single_masks,
                )

        transformer_valid = encoding_mask.unsqueeze(-1).to(self.dtype)
        flat = self.latent_decoder_mean_n_log_scale(single.to(self.dtype)) * transformer_valid
        flat = _restore_token_order(flat, permutation)
        valid = output_encoding_mask.unsqueeze(-1).to(self.dtype)
        flat = flat * valid
        mean, log_scale = torch.chunk(flat, chunks=2, dim=-1)
        if sample:
            if sampling_stream:
                # A caller replaying a reference implementation's random sequence
                # installs a draw this call must take its turn in.
                noise = randn(log_scale.shape, device=log_scale.device, dtype=log_scale.dtype, generator=generator)
            elif generator is None:
                noise = torch.randn_like(log_scale)
            else:
                noise = torch.randn(
                    log_scale.shape, device=log_scale.device, dtype=log_scale.dtype, generator=generator
                )
            latent_pre_mask = mean + noise * torch.exp(log_scale)
        else:
            latent_pre_mask = mean
        latent = latent_pre_mask * valid
        return {
            "mean": mean,
            "log_scale": log_scale,
            "z_latent": latent,
            "z_latent_pre_ln": latent_pre_mask,
            "encoding_mask": output_encoding_mask,
        }
