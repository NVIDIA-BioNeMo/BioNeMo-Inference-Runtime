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

"""Draw noise and coordinate permutations from one caller-owned stream."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_GENERATOR: ContextVar[torch.Generator | None] = ContextVar("bionemo_ir_generator", default=None)


@contextmanager
def use_generator(generator: torch.Generator) -> Iterator[None]:
    """Bind a rollout stream; nested scopes restore it even after exceptions.

    Args:
        generator: Default stream for draws in this scope; an explicit draw generator takes precedence.
    """
    token = _GENERATOR.set(generator)
    try:
        yield
    finally:
        _GENERATOR.reset(token)


def randperm(
    n: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Permute elements using an explicit, scoped, or default generator.

    Args:
        n: Number of elements in the permutation.
        device: Device on which to draw.
        generator: Stream overriding the scoped generator; None uses the scope, then the device default.
    """
    return torch.randperm(n, device=device, generator=generator if generator is not None else _GENERATOR.get())


def randn(
    shape: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw Gaussian noise using an explicit, scoped, or default generator.

    Args:
        shape: Output dimensions.
        device: Device on which to draw.
        dtype: Output dtype; None uses PyTorch's default.
        generator: Stream overriding the scoped generator; None uses the scope, then the device default.
    """
    return torch.randn(
        tuple(shape), device=device, dtype=dtype, generator=generator if generator is not None else _GENERATOR.get()
    )
