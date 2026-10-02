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
"""Variant selection and PyTorch reference for the fused transition MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._config import TransitionMlpVariant, get_tile_params

if TYPE_CHECKING:
    from .cutedsl import TransitionMlpCuTe, TransitionMlpOp

_SUPPORTED_DTYPE = torch.bfloat16
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_transition_mlp_cute_instances: dict[int, TransitionMlpCuTe] = {}


def transition_mlp_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    residual: torch.Tensor | None,
    mask: torch.Tensor | None,
    activation: str = "relu",
) -> torch.Tensor:
    """``[residual +] mask[..., None] * (act(x @ w1.T + b1) @ w2.T + b2)`` in PyTorch.

    ``act`` is ReLU, or for ``"silu_gate"`` ``silu(gate) * value`` where ``w1``'s first half of
    rows produces ``value`` and its second half ``gate``, as in :class:`Transition`. For
    ``"silu_gate_3way"``, ``w1``'s thirds produce ``value``, ``gate`` and ``second`` and ``act`` is
    ``silu(gate) * value * second``, as in a 3-way :class:`ConditionedTransitionBlock`.
    """
    projected = torch.nn.functional.linear(x, w1, b1)
    if activation == "silu_gate":
        value, gate = projected.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * value
    elif activation == "silu_gate_3way":
        value, gate, second = projected.chunk(3, dim=-1)
        hidden = torch.nn.functional.silu(gate) * value * second
    else:
        hidden = torch.relu(projected)
    update = torch.nn.functional.linear(hidden, w2, b2)
    if mask is not None:
        update = update * mask.reshape(*update.shape[:-1], 1).to(update.dtype)
    return update if residual is None else residual + update


def get_transition_mlp_op(
    dtype: torch.dtype | None,
    *,
    width: int,
    hidden: int,
    activation: str = "relu",
    has_bias: bool = True,
    has_mask: bool = True,
    has_residual: bool = True,
) -> TransitionMlpOp | None:
    """Return the fused op when this build ships a kernel for the variant on the current GPU, else ``None``.

    SM90 runs the TMA kernel and SM80, SM86 and SM89 the ``cp.async`` one. The kernel ships as
    source in a private checkout and as a packaged CUBIN otherwise. Callers check
    :meth:`TransitionMlpOp.accepts` before computing ``x``, and the op still returns ``None`` for a
    call it cannot run, so callers keep their own path as the fallback.
    """
    variant = TransitionMlpVariant(activation, has_bias, has_mask, has_residual, width, hidden)
    if dtype != _SUPPORTED_DTYPE or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if get_tile_params(sm, variant) is None:
        return None

    from .cutedsl import TransitionMlpCuTe, TransitionMlpOp

    backend = _transition_mlp_cute_instances.get(sm)
    if backend is None:
        backend = _transition_mlp_cute_instances[sm] = TransitionMlpCuTe(sm_version=sm)
    if not backend.ships(dtype, variant):
        return None
    return TransitionMlpOp(backend, variant)
