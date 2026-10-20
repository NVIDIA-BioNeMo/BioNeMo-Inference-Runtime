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

"""One seam for the random draws a rollout consumes.

Samplers call :func:`randn` instead of ``torch.randn`` so an embedding
application can supply the stream. A host that already owns per-design
generators — one independent stream per batch row, say — installs its own draw
with :func:`normal_draws` and keeps reproducibility across batch layouts that
a single batched generator cannot give.

Noise is not the only consumer. Feature construction permutes coordinate groups,
which draws from the same stream in the reference implementation. :func:`randperm`
is that second seam, and a host that overrides one should override both: a stream
that misses the permutations drifts from the reference's at the first draw after
one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch

#: ``(shape, *, device, dtype) -> Tensor``: an override's draw signature.
NormalDraw = Callable[..., torch.Tensor]

_DRAW: ContextVar[NormalDraw | None] = ContextVar("bionemo_ir_normal_draw", default=None)


@contextmanager
def normal_draws(draw: NormalDraw) -> Iterator[None]:
    """Route this scope's sampler draws through ``draw``.

    ``draw`` owns its own stream, so the ``generator`` the sampler would have
    used is deliberately not forwarded: an override that honoured it could not
    keep a row's stream independent of its batch position.
    """
    token = _DRAW.set(draw)
    try:
        yield
    finally:
        _DRAW.reset(token)


#: ``(n, *, device) -> Tensor``: an override's permutation signature.
Permutation = Callable[..., torch.Tensor]

_PERMUTE: ContextVar[Permutation | None] = ContextVar("bionemo_ir_permutation", default=None)


@contextmanager
def permutations(permute: Permutation) -> Iterator[None]:
    """Route this scope's feature permutations through ``permute``.

    Install it alongside :func:`normal_draws` when the override's stream has to
    track the reference's, which consumes both from one generator.
    """
    token = _PERMUTE.set(permute)
    try:
        yield
    finally:
        _PERMUTE.reset(token)


def randperm(
    n: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Permute ``n`` elements using the active stream."""
    permute = _PERMUTE.get()
    if permute is None:
        return torch.randperm(n, device=device, generator=generator)
    value = permute(n, device=device)
    if tuple(value.shape) != (n,):
        raise ValueError(f"permutation returned shape {tuple(value.shape)}, expected {(n,)}")
    return value.to(device=device)


def randn(
    shape: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw standard normal noise from the active stream."""
    draw = _DRAW.get()
    if draw is None:
        return torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)
    value = draw(tuple(shape), device=device, dtype=dtype)
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"normal draw returned shape {tuple(value.shape)}, expected {tuple(shape)}")
    return value.to(device=device, dtype=dtype)
