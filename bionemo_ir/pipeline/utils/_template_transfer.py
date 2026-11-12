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

"""Producer provenance for transferring populated template slots."""

from typing import Any

import numpy as np
import torch

TEMPLATE_TRANSFER_COLUMN = "_bioir_template_slots"


def record_template_slots(context: dict[str, Any], name: str, tensor: torch.Tensor, active: tuple[int, ...]) -> None:
    """Record slots whose bytes must be copied; every other slot must be +0.

    Only the producer that initializes the tensor may certify this invariant.
    Keep the tensor identity until collation to discard replaced features.
    """
    context.setdefault(TEMPLATE_TRANSFER_COLUMN, {})[name] = (tensor, active)


def attach_template_slots(output: dict[str, Any], context: dict[str, Any], add_batch_dim: bool) -> None:
    """Attach transferable provenance after the final collator adds its batch axis."""
    output.pop(TEMPLATE_TRANSFER_COLUMN, None)
    specs = {}
    axis = int(add_batch_dim)
    for name, (original, active) in context.get(TEMPLATE_TRANSFER_COLUMN, {}).items():
        value = output.get(name)
        if not isinstance(value, torch.Tensor):
            continue
        # Collation only unsqueezes; a replaced tensor must not inherit provenance.
        expected = original.unsqueeze(0) if add_batch_dim else original
        if (
            value.shape != expected.shape
            or value.stride() != expected.stride()
            or value.dtype != expected.dtype
            or value.data_ptr() != original.data_ptr()
        ):
            continue
        if len(active) < value.shape[axis]:
            specs[name] = (axis, active, tuple(value.shape))
    if specs:
        output[TEMPLATE_TRANSFER_COLUMN] = specs


def transfer_template_tensor(
    value: torch.Tensor | np.ndarray,
    device: torch.device | str,
    spec: tuple[int, tuple[int, ...], tuple[int, ...]],
) -> torch.Tensor:
    """Create dense +0 slots on the device and copy populated slots verbatim.

    Args:
        value: Dense CPU feature, with an optional leading batch dimension.
        device: Destination device.
        spec: Template axis, populated slot indices and original logical shape.
    """
    axis, active, shape = spec
    if tuple(value.shape) != tuple(shape) or not 0 <= axis < value.ndim:
        raise ValueError("Template transfer provenance does not match the feature shape")
    if len(set(active)) != len(active) or any(slot < 0 or slot >= shape[axis] for slot in active):
        raise ValueError("Invalid populated template slots")
    if isinstance(value, np.ndarray):
        # Select before copying read-only arrays; no full dense host copy is needed.
        dtype = torch.from_numpy(np.empty(0, dtype=value.dtype)).dtype
    else:
        dtype = value.dtype
    output = torch.zeros(shape, dtype=dtype, device=device)
    for slot in active:
        if isinstance(value, np.ndarray):
            payload = torch.from_numpy(np.take(value, slot, axis=axis))
        else:
            payload = value.select(axis, slot)
        output.select(axis, slot).copy_(payload)
    return output
