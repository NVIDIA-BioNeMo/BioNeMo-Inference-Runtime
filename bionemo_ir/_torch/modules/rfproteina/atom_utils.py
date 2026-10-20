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

"""RFProteína coordinate units, condition handling, and tensor validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

ANGSTROM_TO_NANOMETER = 0.1
NANOMETER_TO_ANGSTROM = 10.0


def angstrom_to_nanometer(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert a coordinate tensor from ångströms to nanometres.

    The operation preserves the tensor's shape, dtype, and device.

    Args:
        coordinates: Coordinates in ångströms.

    Returns:
        Coordinates in nanometres.

    Shape:
        Accepts any shape. Structure-model coordinates are typically
        ``[..., N_atom, 3]`` or ``[B, N_token, N_atom, 3]``.

    Used by:
        RFProteina condition preparation and representation construction.
    """
    return coordinates * ANGSTROM_TO_NANOMETER


def nanometer_to_angstrom(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert a coordinate tensor from nanometres to ångströms.

    The operation preserves the tensor's shape, dtype, and device.

    Args:
        coordinates: Coordinates in nanometres.

    Returns:
        Coordinates in ångströms.

    Shape:
        Accepts any shape. Structure-model coordinates are typically
        ``[..., N_atom, 3]`` or ``[B, N_token, N_atom, 3]``.

    Used by:
        RFProteína coordinate output boundaries. Model outputs currently remain
        in nanometres, so this inverse conversion is primarily a test helper.
    """
    return coordinates * NANOMETER_TO_ANGSTROM


def require_shape(tensor: torch.Tensor, expected: Sequence[int], name: str) -> torch.Tensor:
    """Validate every dimension of a tensor.

    Args:
        tensor: Tensor to validate.
        expected: Required complete shape.
        name: Feature name included in an error message.

    Returns:
        The original tensor, allowing validation inside an expression.

    Raises:
        ValueError: If the tensor shape differs from ``expected``.

    Shape:
        Generic. For example, RFProteina validates token tensors as ``[B, N]``,
        atom masks as ``[B, N, 37]``, and coordinates as ``[B, N, 37, 3]``.

    Used by:
        RFProteina atomized-ligand, motif-insertion, partial-redesign, and
        coordinate-condition paths.
    """
    expected_shape = tuple(expected)
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def require_leading_shape(tensor: torch.Tensor, expected: Sequence[int], name: str) -> torch.Tensor:
    """Validate a tensor prefix while allowing arbitrary trailing dimensions.

    This is useful for features whose batch, token, or pair axes are fixed but
    whose payload dimensions vary.

    Args:
        tensor: Tensor to validate.
        expected: Required leading dimensions.
        name: Feature name included in an error message.

    Returns:
        The original tensor.

    Raises:
        ValueError: If the leading dimensions differ from ``expected``.

    Shape:
        ``tensor`` is ``[*expected, ...]``; only the ``*expected`` prefix is
        checked.

    Used by:
        RFProteina optional token metadata ``[B, N, ...]`` and pair metadata
        ``[B, N, N, ...]`` through :func:`optional_feature`.
    """
    expected_shape = tuple(expected)
    if tensor.shape[: len(expected_shape)] != expected_shape:
        raise ValueError(f"{name} must have leading shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def optional_feature(
    features: Mapping[str, torch.Tensor],
    key: str,
    leading_shape: Sequence[int],
    device: torch.device,
) -> torch.Tensor | None:
    """Read an optional feature, validate its prefix, and move it to a device.

    Missing features remain ``None``. Present features retain their dtype and
    trailing dimensions.

    Args:
        features: Feature mapping to read.
        key: Name of the optional feature.
        leading_shape: Required batch/token/pair prefix.
        device: Destination device for a present feature.

    Returns:
        The device-local feature, or ``None`` when ``key`` is absent.

    Raises:
        ValueError: If a present feature has the wrong leading dimensions.

    Shape:
        Present values must be ``[*leading_shape, ...]``. Current RFProteina
        callers use ``[B, N, ...]`` for token metadata and ``[B, N, N, ...]``
        for ligand bond metadata.

    Used by:
        RFProteina atomized-ligand and DNA/RNA output materializers.
    """
    value = features.get(key)
    if value is None:
        return None
    require_leading_shape(value, leading_shape, key)
    return value.to(device=device)


def finite_atom_mask(coordinates: torch.Tensor) -> torch.Tensor:
    """Mark atoms that carry any finite coordinate component.

    Args:
        coordinates: Coordinate tensor whose last dimension is xyz.

    Returns:
        Boolean mask with the xyz dimension removed.

    Shape:
        ``[..., 3]`` becomes ``[...]``. RFProteina atom37 coordinates are
        ``[B, N, 37, 3]``, so the mask is ``[B, N, 37]``.
    """
    return torch.isfinite(coordinates).any(dim=-1)


def condition_pair(
    features: Mapping[str, torch.Tensor],
    value_key: str,
    mask_key: str,
    *,
    finite_marks_conditioned: bool = False,
    unspecified_token: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Read a condition tensor and its companion mask.

    Args:
        features: Mapping containing model input features.
        value_key: Key for the condition values.
        mask_key: Key selecting valid entries in the condition values.
        finite_marks_conditioned: When the mask is absent, treat an entry as
            conditioned if any value component is finite. Absolute coordinates
            use this.
        unspecified_token: When the mask is absent, treat every other value as
            conditioned. Sequence conditions use the unified mask token.

    Returns:
        ``(values, mask)`` when the values are present, otherwise
        ``(None, None)``. An explicit mask is returned unchanged.

    Raises:
        ValueError: If the mask is present without values, if both derivation
            modes are set, or if values are present without a mask and neither
            derivation mode is set.

    Shape:
        The helper does not impose a shape. RFProteina currently uses sequence
        pairs ``[B, N]``/``[B, N]`` and coordinate pairs
        ``[B, N, 37, 3]``/``[B, N, 37]``.

    Used by:
        RFProteina atomized-ligand, DNA/RNA, motif-insertion, and
        partial-redesign condition handling.
    """
    value = features.get(value_key)
    mask = features.get(mask_key)
    if value is None and mask is None:
        return None, None
    if value is None:
        raise ValueError(f"{value_key} and {mask_key} must be supplied together")
    if mask is None:
        if finite_marks_conditioned == (unspecified_token is not None):
            raise ValueError(f"{value_key} and {mask_key} must be supplied together")
        mask = finite_atom_mask(value) if finite_marks_conditioned else value != unspecified_token
    return value, mask


def prepare_coordinate_condition(
    values_angstrom: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
    *,
    intersect_mask: torch.Tensor | None = None,
    value_name: str = "coordinate condition",
    mask_name: str = "coordinate condition mask",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare externally supplied coordinates for masked model use.

    Condition values must match ``reference`` and are converted from
    ångströms to nanometres, as well as to the reference device and dtype. The
    condition mask can optionally be intersected with an atom-availability
    mask. Only active values are required to be finite.

    Args:
        values_angstrom: Condition coordinates shaped like ``reference``.
        mask: Boolean-compatible mask shaped like ``reference.shape[:-1]``.
        reference: Model coordinate tensor defining shape, device, and dtype.
        intersect_mask: Optional additional mask, usually atom occupancy.
        value_name: Condition name used in validation errors.
        mask_name: Mask name used in validation errors.

    Returns:
        Converted coordinates and the effective boolean mask.

    Raises:
        ValueError: If shapes are incompatible or an active value is non-finite.

    Shape:
        ``reference`` and ``values_angstrom`` are ``[..., 3]``. ``mask`` and
        ``intersect_mask`` are ``[...]``. RFProteina uses
        ``[B, N, 37, 3]`` coordinates and ``[B, N, 37]`` masks.

    Used by:
        RFProteina DNA/RNA and motif-insertion materializers, and indirectly
        the atomized-ligand and partial-redesign materializers through
        :func:`restore_conditioned_coordinates`.
    """
    require_shape(values_angstrom, reference.shape, value_name)
    require_shape(mask, reference.shape[:-1], mask_name)

    values = angstrom_to_nanometer(values_angstrom.to(device=reference.device, dtype=reference.dtype))
    effective_mask = mask.to(device=reference.device, dtype=torch.bool)
    if intersect_mask is not None:
        require_shape(intersect_mask, reference.shape[:-1], "intersect_mask")
        effective_mask = effective_mask & intersect_mask.to(device=reference.device, dtype=torch.bool)
    if torch.any(effective_mask) and not torch.isfinite(values[effective_mask]).all():
        raise ValueError(f"{value_name} contains non-finite conditioned values")
    return values, effective_mask


def restore_conditioned_coordinates(
    generated_coordinates: torch.Tensor,
    values_angstrom: torch.Tensor,
    mask: torch.Tensor,
    *,
    intersect_mask: torch.Tensor | None = None,
    value_name: str = "coordinate condition",
    mask_name: str = "coordinate condition mask",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace generated atom coordinates wherever a condition is active.

    This first applies :func:`prepare_coordinate_condition`, then selects the
    prepared coordinates under the resulting mask. Generated coordinates are
    retained everywhere else.

    Args:
        generated_coordinates: Model coordinates in nanometres.
        values_angstrom: External condition coordinates in ångströms.
        mask: Entries to restore from ``values_angstrom``.
        intersect_mask: Optional additional atom-availability mask.
        value_name: Condition name used in validation errors.
        mask_name: Mask name used in validation errors.

    Returns:
        The restored coordinates and the effective boolean condition mask.

    Shape:
        Inputs and restored coordinates are ``[..., 3]``; masks are ``[...]``.
        RFProteina uses ``[B, N, 37, 3]`` and ``[B, N, 37]`` respectively.

    Used by:
        RFProteina atomized-ligand and partial-redesign output materializers.
    """
    values, effective_mask = prepare_coordinate_condition(
        values_angstrom,
        mask,
        generated_coordinates,
        intersect_mask=intersect_mask,
        value_name=value_name,
        mask_name=mask_name,
    )
    restored = torch.where(effective_mask.unsqueeze(-1), values, generated_coordinates)
    return restored, effective_mask
