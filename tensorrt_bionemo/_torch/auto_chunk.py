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

Many layers have an activation whose peak size grows with a problem dimension (``N`` residues,
``S`` sequences). When it blows up, and the op is *position-wise* along an output dimension --
``f(cat([a, b], dim)) == cat([f(a), f(b)], dim)`` -- we can slice that dimension, run ``f`` per
slice, and ``cat`` the outputs (see :func:`chunk_apply`). This is bit-identical to the dense call
but bounds any internal ``[.., dim, ..]`` activation to ``chunk_size`` rows. Examples:

* ``transition_z`` (pair FFN): row-chunk dim=1 -> ``[N, N, 2*hidden]`` shrinks to ``[chunk, N, ...]``.
* ``OuterProductMean``: row-chunk the output token dim -> the dominant ``[N, N, c_hidden**2]``
  einsum shrinks to ``[chunk, N, c_hidden**2]`` (the key/``c_hidden`` dims stay whole).
* ``PairWeightedAveraging``: row-chunk the sequence dim ``S`` -> the ``[H, S, N, D]`` temporaries
  shrink to ``[H, chunk, N, D]`` (its attention mixes only the token dims, so ``S`` is row-safe).

All engage only above a size threshold, so small problems stay on the dense path untouched.

Registry
--------
Rather than hardcode thresholds per layer, chunkable layers look up a shared, centrally-tunable
:class:`ChunkPolicy` from the global :data:`CHUNK_REGISTRY` by name::

    from tensorrt_bionemo._torch.auto_chunk import CHUNK_REGISTRY, PAIR_WEIGHTED_AVERAGING
    policy = CHUNK_REGISTRY.get(PAIR_WEIGHTED_AVERAGING)

Policies can be re-tuned globally at runtime (e.g. to disable chunking or change the threshold)::

    CHUNK_REGISTRY.set(PAIR_WEIGHTED_AVERAGING, min_size=4096)
    CHUNK_REGISTRY.disable(OUTER_PRODUCT_MEAN)

A layer may also be constructed with an explicit ``chunk_policy=`` to override the registry default.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import torch

# Chunk granularity for the pair feed-forward (``transition_z``): 512 rows per slice.
DEFAULT_PAIR_CHUNK_ROWS = 512

# The autochunk trigger threshold (residues/tokens above which we chunk) is *memory-scaled*: a fixed
# value is either too eager on large GPUs (needless chunk overhead) or too late on small ones (OOM
# before chunking engages). Pair activations are O(N^2), so the largest N that fits before chunking
# scales ~sqrt(total_memory); we anchor at a reference GPU and scale to the actual device.
_REFERENCE_GPU_GB = 80.0
DEFAULT_AUTOCHUNK_MIN_REF = 2560  # threshold anchor for an 80 GB GPU
_AUTOCHUNK_MIN_FLOOR = 512
# Sentinel ``min_size`` meaning "resolve from GPU memory at runtime". Resolution is deferred to the
# forward pass (not import) so importing this module never initializes a CUDA context.
AUTOCHUNK_MIN_AUTO = -1


@functools.lru_cache(maxsize=8)
def default_autochunk_min(device=None) -> int:
    """Memory-scaled autochunk threshold for ``device`` (current CUDA device when ``None``).

    Anchored at :data:`DEFAULT_AUTOCHUNK_MIN_REF` residues on a :data:`_REFERENCE_GPU_GB` GPU and
    scaled by ``sqrt(total_memory / reference)`` (pair activations are O(N^2)); rounded to a multiple
    of 128 and floored at :data:`_AUTOCHUNK_MIN_FLOOR`. Falls back to the reference when CUDA is
    unavailable. Cached, so each device is queried at most once.
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


# Back-compat alias for the unscaled reference anchor (prefer ``default_autochunk_min()``).
DEFAULT_AUTOCHUNK_MIN = DEFAULT_AUTOCHUNK_MIN_REF

# Registry keys for the known chunkable ops (use these instead of bare strings).
PAIR_TRANSITION = "pair_transition"
DIFFUSION_PAIR_TRANSITION = "diffusion_pair_transition"
MSA_TRANSITION = "msa_transition"
PAIR_WEIGHTED_AVERAGING = "pair_weighted_averaging"
OUTER_PRODUCT_MEAN = "outer_product_mean"
TRIANGLE_ATTENTION = "triangle_attention"
CONTACT_PROB = "contact_prob"

# OSS Protenix ``MSAStack.msa_chunk_size`` default: chunk MSA rows (dim S of
# ``[B, S, N, C_m]``) so the SwiGLU ``[S, N, 2*hidden]`` transient stays bounded.
DEFAULT_MSA_CHUNK_ROWS = 2048
# Engage MSA-row chunking once S exceeds this (independent of the N-residue
# memory-scaled pair threshold — deep MSAs OOMs at modest N, e.g. H1185).
DEFAULT_MSA_AUTOCHUNK_MIN = 2048


def iter_chunks(total: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, length)`` covering ``range(total)`` in steps of ``chunk`` (last may be short)."""
    for start in range(0, total, chunk):
        yield start, min(chunk, total - start)


@dataclass(frozen=True)
class ChunkPolicy:
    """Describes *when* and *how much* to chunk a chunkable op.

    The meaning of :attr:`chunk_size` is per-op (rows for a position-wise concat op, heads or
    hidden-channels for a reduce op) -- see the registration site. The trigger threshold
    :attr:`min_size` is compared against whatever "problem size" the op reports (e.g. ``N`` residues),
    which need not be the chunked dimension itself.

    Attributes:
        chunk_size: elements per chunk along the op's chunked dimension (``<= 0`` disables chunking).
        min_size: only chunk when the reported problem size exceeds this (keeps small problems dense).
            The default (:data:`AUTOCHUNK_MIN_AUTO`) resolves lazily to a GPU-memory-scaled threshold
            (see :func:`default_autochunk_min`); set an explicit ``>= 0`` value to pin it.
        dim: for :func:`chunk_apply` (concat style), the dim to slice inputs / concat outputs along.
        min_rank: for :meth:`should_chunk`, only chunk tensors with at least this many dims. Defaults
            to ``4`` so pair activations ``[B, N, N, C]`` chunk while cheap single-rep ``[B, N, C]``
            do not. Set to ``0`` to disable the rank gate.
        enabled: master switch.
    """

    chunk_size: int = DEFAULT_PAIR_CHUNK_ROWS
    min_size: int = AUTOCHUNK_MIN_AUTO
    dim: int = 1
    min_rank: int = 4
    enabled: bool = True

    def resolved_min_size(self, device=None) -> int:
        """Effective threshold: explicit ``min_size`` when set (``>= 0``), else the memory-scaled default."""
        return self.min_size if self.min_size >= 0 else default_autochunk_min(device)

    def should_chunk_size(self, n: int, device=None) -> bool:
        """True iff a problem of size ``n`` is large enough (and this policy is on) to chunk."""
        return self.enabled and self.chunk_size > 0 and n > self.resolved_min_size(device)

    def should_chunk(self, x: torch.Tensor) -> bool:
        """Tensor form of :meth:`should_chunk_size` for concat-style ops: gates on rank + ``dim`` extent."""
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
        """Return a copy with selected fields overridden (e.g. ``policy.replace(enabled=False)``)."""
        return replace(self, **changes)


class ChunkRegistry:
    """Central name -> :class:`ChunkPolicy` map for chunkable layers.

    Lets layers look up a shared, centrally-tunable policy instead of hardcoding thresholds, and
    lets callers retune globally at runtime.
    """

    def __init__(self) -> None:
        self._policies: dict[str, ChunkPolicy] = {}

    def register(self, name: str, policy: ChunkPolicy, *, overwrite: bool = True) -> ChunkPolicy:
        """Register ``policy`` under ``name`` (default overwrites; pass ``overwrite=False`` to keep)."""
        if not overwrite and name in self._policies:
            return self._policies[name]
        self._policies[name] = policy
        return policy

    def get(self, name: str, default: ChunkPolicy | None = None) -> ChunkPolicy | None:
        """Return the policy for ``name`` (or ``default`` if unregistered)."""
        return self._policies.get(name, default)

    def set(self, name: str, **overrides) -> ChunkPolicy:
        """Override selected fields of ``name``'s policy in place (creating it from defaults if new)."""
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


# Global registry + built-in defaults for the known chunkable ops.
CHUNK_REGISTRY = ChunkRegistry()

# All chunk output rows along dim=1 (concat), and leave ``min_size`` at AUTOCHUNK_MIN_AUTO -> the
# GPU-memory-scaled threshold. ``chunk_size`` is rows-per-slice; ``min_rank=4`` keeps cheap rank-3
# activations (e.g. single-rep transition_s) on the dense path.
# transition_z: pair row dim (N).
CHUNK_REGISTRY.register(PAIR_TRANSITION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
# Diffusion pair-conditioning transition_z: same row-chunk semantics as trunk
# transition_z ([B, N, N, C], dim=1), but a separate registry key so diffusion
# thresholds can be tuned independently. Single-conditioning transition_s stays
# dense (its dim=1 is the sample axis).
CHUNK_REGISTRY.register(DIFFUSION_PAIR_TRANSITION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
# MSA Transition (SwiGLU on ``[B, S, N, C_m]``): chunk the MSA-row dim S, matching
# OSS ``MSAStack.inference_forward`` (msa_chunk_size=2048). Without this, deep
# MSAs at moderate N (CASP15 H1185: S≈35k, N≈1332) allocate ~45 GB for the fused
# ``2*hidden`` projection and OOM on 80 GB while OSS fits.
CHUNK_REGISTRY.register(
    MSA_TRANSITION,
    ChunkPolicy(chunk_size=DEFAULT_MSA_CHUNK_ROWS, min_size=DEFAULT_MSA_AUTOCHUNK_MIN, dim=1, min_rank=4),
)
# PairWeightedAveraging: sequence dim S (the einsum's non-token, row-safe axis).
CHUNK_REGISTRY.register(PAIR_WEIGHTED_AVERAGING, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
# OuterProductMean: output token-row dim. Fewer rows/chunk since its intermediate carries an extra
# c_hidden**2 factor ([chunk, N, c_hidden**2]).
CHUNK_REGISTRY.register(OUTER_PRODUCT_MEAN, ChunkPolicy(chunk_size=128, dim=1, min_rank=4))
# TriangleAttentionNode: query-row dim I (attention over the key dim J is independent per row).
# Disabled by default -- the flash-attention triangle kernels already bound memory; enable via
# ``CHUNK_REGISTRY.enable(TRIANGLE_ATTENTION)`` when falling back to a memory-heavy attention backend.
CHUNK_REGISTRY.register(
    TRIANGLE_ATTENTION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4, enabled=False)
)
# Distogram -> contact-probability reduction: pair row dim of ``[B, N, N, num_bins]``. The softmax is
# per-row over the bin dim, so row-chunking bounds it to ``[chunk, N, num_bins]`` without changing
# the result.
CHUNK_REGISTRY.register(CONTACT_PROB, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))

# Back-compat alias for the pair-transition default policy.
DEFAULT_PAIR_TRANSITION_POLICY = CHUNK_REGISTRY.get(PAIR_TRANSITION)


def chunk_apply[T](
    fn: Callable[..., T],
    *chunked: torch.Tensor | None,
    policy: ChunkPolicy | None = None,
    cat_dim: int | None = None,
    **passthrough,
) -> T:
    """Evaluate ``fn(*chunked, **passthrough)`` in row-slices along ``policy.dim`` and concatenate.

    For *concat-style* (position-wise) ops only: output row ``i`` must depend only on input row ``i``.

    Args:
        fn: position-wise callable, invoked as ``fn(*sliced, **passthrough)`` per slice.
        *chunked: tensors sliced in lockstep along ``policy.dim``. Entries that are ``None``, not
            tensors, or whose ``dim`` extent differs from the primary (first) tensor are forwarded
            unsliced -- e.g. an already-reduced bias, a broadcast scalar, or ``mask=None``.
        policy: chunking policy (defaults to the ``pair_transition`` registry policy).
        cat_dim: dimension to concatenate outputs along (defaults to ``policy.dim``).
        **passthrough: forwarded unchanged to every ``fn`` call (e.g. ``attn_metadata``).

    Returns:
        The same result as a single dense ``fn`` call, assembled from the per-slice outputs.
        ``tuple``/``list`` outputs are concatenated element-wise.
    """
    if policy is None:
        policy = CHUNK_REGISTRY.get(PAIR_TRANSITION)
    primary = chunked[0] if chunked else None
    if primary is None or policy is None or not policy.should_chunk(primary):
        # Dense fast path: identical result and graph, no python-loop / concat overhead.
        return fn(*chunked, **passthrough)

    dim = policy.dim
    out_dim = policy.dim if cat_dim is None else cat_dim
    n = primary.shape[dim]

    def _slice(t: torch.Tensor | None, start: int, length: int):
        # Only slice tensors aligned to the primary along ``dim``; pass everything else
        # (None, scalars, already-reduced biases) through untouched.
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
