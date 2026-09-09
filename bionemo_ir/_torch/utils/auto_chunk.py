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
"""Registry-based output-row chunking for memory-bounded eager execution.

For position-wise ops along an output dim
(``f(cat([a, b], dim)) == cat([f(a), f(b)], dim)``), slice that dim, run ``f``
per slice, and ``cat`` the results (see :func:`chunk_apply`). Peak internal
activations are bounded to ``chunk_size`` rows; the dense result is unchanged.
Chunking engages only above a size threshold.

Layers look up a shared :class:`ChunkPolicy` from :data:`CHUNK_REGISTRY`::

    from bionemo_ir._torch.utils import CHUNK_REGISTRY, PAIR_WEIGHTED_AVERAGING
    policy = CHUNK_REGISTRY.get(PAIR_WEIGHTED_AVERAGING)

    CHUNK_REGISTRY.set(PAIR_WEIGHTED_AVERAGING, min_size=4096)
    CHUNK_REGISTRY.disable(OUTER_PRODUCT_MEAN)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import torch

DEFAULT_PAIR_CHUNK_ROWS = 512

# Threshold scales with sqrt(GPU memory): pair activations are O(N^2).
_REFERENCE_GPU_GB = 80.0
DEFAULT_AUTOCHUNK_MIN_REF = 2560  # residues on an 80 GB GPU
_AUTOCHUNK_MIN_FLOOR = 512
# Sentinel: resolve ``min_size`` from GPU memory at forward time (not import).
AUTOCHUNK_MIN_AUTO = -1


@functools.lru_cache(maxsize=8)
def default_autochunk_min(device=None) -> int:
    """Memory-scaled autochunk threshold for ``device`` (current CUDA device when ``None``).

    Anchored at :data:`DEFAULT_AUTOCHUNK_MIN_REF` on a :data:`_REFERENCE_GPU_GB` GPU,
    scaled by ``sqrt(total_memory / reference)``, rounded to a multiple of 128, and
    floored at :data:`_AUTOCHUNK_MIN_FLOOR`. Falls back to the reference when CUDA
    is unavailable.
    """
    try:
        if not torch.cuda.is_available():
            return DEFAULT_AUTOCHUNK_MIN_REF
        if device is None:
            device = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        scaled = DEFAULT_AUTOCHUNK_MIN_REF * math.sqrt(total_gb / _REFERENCE_GPU_GB)
        return max(_AUTOCHUNK_MIN_FLOOR, int(round(scaled / 128) * 128))
    except Exception:
        return DEFAULT_AUTOCHUNK_MIN_REF


# Back-compat alias (prefer ``default_autochunk_min()``).
DEFAULT_AUTOCHUNK_MIN = DEFAULT_AUTOCHUNK_MIN_REF

# Registry keys for known chunkable ops.
PAIR_TRANSITION = "pair_transition"
DIFFUSION_PAIR_TRANSITION = "diffusion_pair_transition"
MSA_TRANSITION = "msa_transition"
PAIR_WEIGHTED_AVERAGING = "pair_weighted_averaging"
OUTER_PRODUCT_MEAN = "outer_product_mean"
TRIANGLE_ATTENTION = "triangle_attention"
CONTACT_PROB = "contact_prob"

# MSA-row chunking (dim S of ``[B, S, N, C_m]``); independent of the N-residue pair threshold.
DEFAULT_MSA_CHUNK_ROWS = 2048
DEFAULT_MSA_AUTOCHUNK_MIN = 2048


def iter_chunks(total: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, length)`` covering ``range(total)`` in steps of ``chunk``."""
    for start in range(0, total, chunk):
        yield start, min(chunk, total - start)


@dataclass(frozen=True)
class ChunkPolicy:
    """When and how much to chunk a chunkable op.

    Attributes:
        chunk_size: Elements per chunk along the op's chunked dimension
            (``<= 0`` disables chunking).
        min_size: Chunk only when the reported problem size exceeds this.
            :data:`AUTOCHUNK_MIN_AUTO` resolves lazily via :func:`default_autochunk_min`.
        dim: Dimension to slice / concat for :func:`chunk_apply`.
        min_rank: Minimum tensor rank to chunk (default ``4`` skips rank-3 singles).
        enabled: Master switch.
    """

    chunk_size: int = DEFAULT_PAIR_CHUNK_ROWS
    min_size: int = AUTOCHUNK_MIN_AUTO
    dim: int = 1
    min_rank: int = 4
    enabled: bool = True

    def resolved_min_size(self, device=None) -> int:
        """Explicit ``min_size`` when ``>= 0``, else the memory-scaled default."""
        return self.min_size if self.min_size >= 0 else default_autochunk_min(device)

    def should_chunk_size(self, n: int, device=None) -> bool:
        """True if a problem of size ``n`` should be chunked under this policy."""
        return self.enabled and self.chunk_size > 0 and n > self.resolved_min_size(device)

    def should_chunk(self, x: torch.Tensor) -> bool:
        """Like :meth:`should_chunk_size`, gating on tensor rank and ``dim`` extent."""
        if not (
            self.enabled
            and self.chunk_size > 0
            and torch.is_tensor(x)
            and x.dim() >= self.min_rank
            and x.dim() > self.dim
        ):
            return False
        device = x.device if x.is_cuda else None
        return x.shape[self.dim] > self.resolved_min_size(device)

    def replace(self, **changes) -> ChunkPolicy:
        """Return a copy with selected fields overridden."""
        return replace(self, **changes)


class ChunkRegistry:
    """Name -> :class:`ChunkPolicy` map for chunkable layers."""

    def __init__(self) -> None:
        self._policies: dict[str, ChunkPolicy] = {}

    def register(self, name: str, policy: ChunkPolicy, *, overwrite: bool = True) -> ChunkPolicy:
        """Register ``policy`` under ``name``."""
        if not overwrite and name in self._policies:
            return self._policies[name]
        self._policies[name] = policy
        return policy

    def get(self, name: str, default: ChunkPolicy | None = None) -> ChunkPolicy | None:
        """Return the policy for ``name``, or ``default`` if unregistered."""
        return self._policies.get(name, default)

    def set(self, name: str, **overrides) -> ChunkPolicy:
        """Override fields of ``name``'s policy (creates from defaults if missing)."""
        base = self._policies.get(name) or ChunkPolicy()
        policy = base.replace(**overrides)
        self._policies[name] = policy
        return policy

    def enable(self, name: str) -> ChunkPolicy:
        return self.set(name, enabled=True)

    def disable(self, name: str) -> ChunkPolicy:
        return self.set(name, enabled=False)

    def names(self) -> list[str]:
        return list(self._policies)

    def __contains__(self, name: str) -> bool:
        return name in self._policies

    def __getitem__(self, name: str) -> ChunkPolicy:
        return self._policies[name]


CHUNK_REGISTRY = ChunkRegistry()

# Defaults: chunk along dim=1; ``min_size`` uses the memory-scaled threshold unless noted.
CHUNK_REGISTRY.register(PAIR_TRANSITION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
CHUNK_REGISTRY.register(DIFFUSION_PAIR_TRANSITION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
CHUNK_REGISTRY.register(
    MSA_TRANSITION,
    ChunkPolicy(chunk_size=DEFAULT_MSA_CHUNK_ROWS, min_size=DEFAULT_MSA_AUTOCHUNK_MIN, dim=1, min_rank=4),
)
CHUNK_REGISTRY.register(PAIR_WEIGHTED_AVERAGING, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
# Smaller chunks: intermediate carries an extra ``c_hidden**2`` factor.
CHUNK_REGISTRY.register(OUTER_PRODUCT_MEAN, ChunkPolicy(chunk_size=128, dim=1, min_rank=4))
# Off by default; flash triangle kernels already bound memory.
CHUNK_REGISTRY.register(
    TRIANGLE_ATTENTION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4, enabled=False)
)
CHUNK_REGISTRY.register(CONTACT_PROB, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))

DEFAULT_PAIR_TRANSITION_POLICY = CHUNK_REGISTRY.get(PAIR_TRANSITION)


def chunk_apply[T](
    fn: Callable[..., T],
    *chunked: torch.Tensor | None,
    policy: ChunkPolicy | None = None,
    cat_dim: int | None = None,
    **passthrough,
) -> T:
    """Evaluate ``fn(*chunked, **passthrough)`` in row-slices along ``policy.dim`` and concatenate.

    For concat-style (position-wise) ops only: output row ``i`` must depend only on input row ``i``.

    Args:
        fn: Position-wise callable, invoked as ``fn(*sliced, **passthrough)`` per slice.
        *chunked: Tensors sliced in lockstep along ``policy.dim``. ``None``, non-tensors, or
            tensors whose ``dim`` extent differs from the primary are forwarded unsliced.
        policy: Chunking policy (defaults to the ``pair_transition`` registry policy).
        cat_dim: Dimension to concatenate outputs along (defaults to ``policy.dim``).
        **passthrough: Forwarded unchanged to every ``fn`` call.

    Returns:
        Same result as a dense ``fn`` call. ``tuple``/``list`` outputs are concatenated
        element-wise.
    """
    if policy is None:
        policy = CHUNK_REGISTRY.get(PAIR_TRANSITION)
    primary = chunked[0] if chunked else None
    if primary is None or policy is None or not policy.should_chunk(primary):
        return fn(*chunked, **passthrough)

    dim = policy.dim
    out_dim = policy.dim if cat_dim is None else cat_dim
    n = primary.shape[dim]

    def _slice(t: torch.Tensor | None, start: int, length: int):
        if t is None or not torch.is_tensor(t) or t.dim() <= dim or t.shape[dim] != n:
            return t
        return t.narrow(dim, start, length)

    outs = []
    for start, length in iter_chunks(n, policy.chunk_size):
        outs.append(fn(*[_slice(t, start, length) for t in chunked], **passthrough))

    first = outs[0]
    if isinstance(first, (tuple, list)):
        catted = [torch.cat([o[i] for o in outs], dim=out_dim) for i in range(len(first))]
        return type(first)(catted)
    return torch.cat(outs, dim=out_dim)
