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
"""Op and PyTorch reference for the conditioned transition MLP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _config
from .cutedsl import ConditionedTransitionMlpCuTe

__all__ = [
    "EPS",
    "ConditionedTransitionMlpOp",
    "conditioned_transition_mlp_reference",
    "get_conditioned_transition_mlp_op",
]

#: The LayerNorm epsilon the kernel applies.
EPS = 1e-5
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_instances: dict[int, ConditionedTransitionMlpCuTe] = {}


def _rows_aligned(tensor: torch.Tensor) -> bool:
    """Whether ``tensor``'s rows are contiguous and 16-byte aligned, as the kernel's tensor maps need."""
    return tensor.stride(-1) == 1 and tensor.data_ptr() % 16 == 0 and (tensor.stride(0) * 2) % 16 == 0


def conditioned_transition_mlp_reference(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    cond: torch.Tensor,
    eps: float = EPS,
    activation: str = "silu_gate",
) -> torch.Tensor:
    """fp32 ``a + sigmoid(g) * MLP(sigmoid(gamma) * LayerNorm(a) + beta)``, the op's contract.

    ``cond`` row ``r % A`` holds ``[sigmoid(gamma), beta, sigmoid(g)]`` for row ``r`` of ``a``'s flattened rows. MLP is
    the SwiGLU ``(silu(u Wg^T) * (u Wv^T)) W2^T`` with ``w1`` holding the value rows, then the gate rows; the 3-way
    SwiGLU (``"silu_gate_3way"``) multiplies in ``u Ws^T`` from a third block of second-value rows.
    """
    width = a.shape[-1]
    rows = a.reshape(-1, width).float()
    scale, shift, gate = cond.float().view(cond.shape[0], 3, width).unbind(1)
    index = torch.arange(rows.shape[0], device=a.device) % cond.shape[0]
    u = scale[index] * torch.nn.functional.layer_norm(rows, (width,), eps=eps) + shift[index]
    blocks = (u @ w1.float().T).chunk(_config.W1_BLOCKS[activation], dim=-1)
    hidden = torch.nn.functional.silu(blocks[1]) * blocks[0]
    if activation == "silu_gate_3way":
        hidden = hidden * blocks[2]
    mlp = hidden @ w2.float().T
    return (rows + gate[index] * mlp).reshape(a.shape)


@dataclass(frozen=True)
class ConditionedTransitionMlpOp:
    """The conditioned transition MLP bound to one ``(width, hidden, activation)``; see
    :func:`get_conditioned_transition_mlp_op`."""

    backend: ConditionedTransitionMlpCuTe
    width: int
    hidden: int
    activation: str = "silu_gate"

    def __call__(
        self, a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, cond: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """``a + sigmoid(g) * MLP(sigmoid(gamma) * LayerNorm(a) + beta)`` (:func:`conditioned_transition_mlp_reference`),
        or ``None`` for a call this build cannot run.

        Args:
            a: ``[..., width]`` contiguous bf16 input, also the residual; row ``r`` of its flattened rows takes
                ``cond[r % A]``.
            w1: ``[2 * hidden, width]``: the SwiGLU's value rows, then its gate rows; for the 3-way SwiGLU
                ``[3 * hidden, width]``, with its second-value rows last.
            w2: ``[width, hidden]``.
            cond: ``[A, 3 * width]`` per-atom ``[sigmoid(gamma), beta, sigmoid(g)]``, ``A`` dividing ``a``'s rows.
            out: Contiguous output shaped like ``a``; a new tensor by default.
        """
        width, hidden = self.width, self.hidden
        # Device indices rather than torch.device objects, which cost more than the rest of the check.
        device = a.get_device()
        if (
            device < 0
            or a.dtype != torch.bfloat16
            or a.shape[-1] != width
            or not a.is_contiguous()
            or cond.dim() != 2
            or cond.shape[1] != 3 * width
            or w1.shape != (_config.W1_BLOCKS[self.activation] * hidden, width)
            or w2.shape != (width, hidden)
            or any(t.dtype != a.dtype or t.get_device() != device or not t.is_contiguous() for t in (w1, w2, cond))
        ):
            return None
        rows, atoms = a.numel() // width, cond.shape[0]
        if rows == 0 or atoms == 0 or rows % atoms:
            return None
        if out is None:
            out = torch.empty_like(a)
        elif out.shape != a.shape or out.dtype != a.dtype or out.get_device() != device or not out.is_contiguous():
            return None
        a_rows, out_rows = a.view(rows, width), out.view(rows, width)
        if not all(_rows_aligned(t) for t in (a_rows, out_rows, cond)):
            return None
        return out if self.backend.run(a_rows, w1, w2, cond, out_rows, width, hidden, self.activation) else None


def get_conditioned_transition_mlp_op(
    dtype: torch.dtype | None, width: int, hidden: int, activation: str, eps: float, norm_type: str
) -> ConditionedTransitionMlpOp | None:
    """The fused conditioned transition for a block of this shape on this GPU, or ``None``.

    It serves a bf16 block with a 2-way or 3-way SwiGLU (``"silu_gate"``, ``"silu_gate_3way"``) and a LayerNorm AdaLN of
    epsilon :data:`EPS`, at a ``(width, hidden, activation)`` the SM's configs ship (:func:`._config.shipped_shapes`),
    from source in a private checkout and as packaged CUBINs otherwise.
    """
    if (
        dtype != torch.bfloat16
        or activation not in _config.ACTIVATIONS
        or eps != EPS
        or norm_type != "layer_norm"
        or not torch.cuda.is_available()
    ):
        return None
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    if (width, hidden, activation) not in _config.shipped_shapes(sm_version):
        return None
    backend = _instances.get(sm_version)
    if backend is None:
        backend = _instances.setdefault(sm_version, ConditionedTransitionMlpCuTe(sm_version))
    if not backend.ships(width, hidden, activation):
        return None
    return ConditionedTransitionMlpOp(backend, width, hidden, activation)
