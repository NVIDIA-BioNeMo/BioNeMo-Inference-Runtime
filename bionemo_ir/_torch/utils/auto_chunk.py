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
(``f(cat([a, b], dim)) == cat([f(a), f(b)], dim)``), slice that dim and run
``f`` per slice (see :func:`chunk_apply`). Inference writes slices directly
into a preallocated result. Peak internal activations are bounded to ``chunk_size``
rows; the dense result is unchanged. Chunking engages only above a size
threshold.

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

# A [1, 4096, 4096, 128] recycled pair contains exactly 2**31
# elements. Keep the baseline whole-tensor statement through that indexing
# boundary; the owned row update is reserved for the larger capacity regime.
RECYCLE_PAIR_UPDATE_MIN_SIZE = 4096

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
DIFFUSION_CONDITIONING_PROJECTION = "diffusion_conditioning_projection"
INPUT_EMBEDDER_PAIR_BUILD = "input_embedder_pair_build"
RECYCLE_PAIR_UPDATE = "recycle_pair_update"
CONFIDENCE_PAIR_EMBEDDING = "confidence_pair_embedding"
CONFIDENCE_PAIR_PROJECTION = "confidence_pair_projection"
CONFIDENCE_TRIANGLE_ATTENTION = "confidence_triangle_attention"
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
    DIFFUSION_CONDITIONING_PROJECTION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=2, min_rank=5)
)
CHUNK_REGISTRY.register(INPUT_EMBEDDER_PAIR_BUILD, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
CHUNK_REGISTRY.register(
    RECYCLE_PAIR_UPDATE,
    ChunkPolicy(
        chunk_size=DEFAULT_PAIR_CHUNK_ROWS,
        min_size=RECYCLE_PAIR_UPDATE_MIN_SIZE,
        dim=1,
        min_rank=4,
    ),
)
CHUNK_REGISTRY.register(CONFIDENCE_PAIR_EMBEDDING, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
CHUNK_REGISTRY.register(CONFIDENCE_PAIR_PROJECTION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=2, min_rank=5))
CHUNK_REGISTRY.register(
    CONFIDENCE_TRIANGLE_ATTENTION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4)
)
CHUNK_REGISTRY.register(
    MSA_TRANSITION,
    ChunkPolicy(chunk_size=DEFAULT_MSA_CHUNK_ROWS, min_size=DEFAULT_MSA_AUTOCHUNK_MIN, dim=1, min_rank=4),
)
CHUNK_REGISTRY.register(PAIR_WEIGHTED_AVERAGING, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
# Smaller chunks: intermediate carries an extra ``c_hidden**2`` factor.
CHUNK_REGISTRY.register(OUTER_PRODUCT_MEAN, ChunkPolicy(chunk_size=128, dim=1, min_rank=4))
# Bound QKV and output temporaries for long pair representations. The dense
# fast path remains active below the memory-scaled threshold.
CHUNK_REGISTRY.register(TRIANGLE_ATTENTION, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))
CHUNK_REGISTRY.register(CONTACT_PROB, ChunkPolicy(chunk_size=DEFAULT_PAIR_CHUNK_ROWS, dim=1, min_rank=4))

DEFAULT_PAIR_TRANSITION_POLICY = CHUNK_REGISTRY.get(PAIR_TRANSITION)


def chunk_apply[T](
    fn: Callable[..., T],
    *chunked: torch.Tensor | None,
    policy: ChunkPolicy | None = None,
    cat_dim: int | None = None,
    **passthrough,
) -> T:
    """Evaluate ``fn(*chunked, **passthrough)`` in row-slices along ``policy.dim`` and assemble.

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

    chunks = iter(iter_chunks(n, policy.chunk_size))
    first_start, first_length = next(chunks)
    first = fn(*[_slice(t, first_start, first_length) for t in chunked], **passthrough)

    container_type = type(first) if isinstance(first, (tuple, list)) else None
    output_arity = len(first) if container_type is not None else 1

    def _outputs(value: T) -> list[torch.Tensor]:
        if container_type is None:
            if not torch.is_tensor(value):
                raise TypeError("chunk outputs must be tensors or a tuple/list of tensors")
            return [value]
        if not isinstance(value, container_type) or len(value) != output_arity:
            raise TypeError("chunk outputs must retain their tuple/list type and length")
        if not all(torch.is_tensor(output) for output in value):
            raise TypeError("chunk output tuples/lists must contain only tensors")
        return list(value)

    first_outputs = _outputs(first)

    def _normalize_dim(output: torch.Tensor) -> int:
        normalized_dim = out_dim if out_dim >= 0 else output.dim() + out_dim
        if not 0 <= normalized_dim < output.dim():
            raise IndexError(f"cat_dim {out_dim} is out of range for a rank-{output.dim()} output")
        return normalized_dim

    result_dims = [_normalize_dim(output) for output in first_outputs]
    reference_shapes = [tuple(output.shape) for output in first_outputs]

    def _validate_shapes(outputs: list[torch.Tensor]) -> None:
        for output, result_dim, reference_shape in zip(outputs, result_dims, reference_shapes, strict=True):
            if output.dim() != len(reference_shape):
                raise ValueError(
                    f"chunk output rank {output.dim()} does not match the first output rank {len(reference_shape)}"
                )
            for axis, (extent, reference_extent) in enumerate(zip(output.shape, reference_shape, strict=True)):
                if axis != result_dim and extent != reference_extent:
                    raise ValueError(
                        f"chunk output extent {extent} along non-concatenated dim {axis} "
                        f"does not match the first output extent {reference_extent}"
                    )

    def _can_preallocate(output: torch.Tensor, result_dim: int, chunk_length: int) -> bool:
        # ``new_empty`` matches torch.cat's layout only for standard contiguous
        # tensors. Preserve the general concat contract for row-expanding and
        # alternate-memory-format outputs by retaining the original cat path.
        return output.layout == torch.strided and output.is_contiguous() and output.shape[result_dim] == chunk_length

    def _pack(outputs: list[torch.Tensor]) -> T:
        if container_type is None:
            return outputs[0]
        return container_type(outputs)

    def _concatenate(
        collected: list[list[torch.Tensor]],
        remaining_chunks: Iterator[tuple[int, int]],
    ) -> T:
        for start, length in remaining_chunks:
            current = fn(*[_slice(t, start, length) for t in chunked], **passthrough)
            current_outputs = _outputs(current)
            _validate_shapes(current_outputs)
            for values, output in zip(collected, current_outputs, strict=True):
                values.append(output)
        return _pack(
            [torch.cat(outputs, dim=result_dim) for outputs, result_dim in zip(collected, result_dims, strict=True)]
        )

    _validate_shapes(first_outputs)
    if not all(
        _can_preallocate(output, result_dim, first_length)
        for output, result_dim in zip(first_outputs, result_dims, strict=True)
    ):
        return _concatenate([[output] for output in first_outputs], chunks)

    # Copy each compatible inference chunk into final storage as it is
    # produced. A list followed by ``torch.cat`` retains every chunk and then
    # allocates a second full output.
    def _allocate(output: torch.Tensor, result_dim: int, chunk_length: int) -> torch.Tensor:
        shape = list(output.shape)
        shape[result_dim] = n
        result = output.new_empty(shape)
        result.narrow(result_dim, first_start, chunk_length).copy_(output)
        return result

    results = [
        _allocate(output, result_dim, first_length)
        for output, result_dim in zip(first_outputs, result_dims, strict=True)
    ]
    written_chunks = [(first_start, first_length)]
    del first_outputs, first

    for start, length in chunks:
        current = fn(*[_slice(t, start, length) for t in chunked], **passthrough)
        current_outputs = _outputs(current)
        _validate_shapes(current_outputs)
        compatible = all(
            _can_preallocate(output, result_dim, length)
            and output.dtype == result.dtype
            and output.device == result.device
            for output, result, result_dim in zip(current_outputs, results, result_dims, strict=True)
        )
        if not compatible:
            collected = [
                [
                    result.narrow(result_dim, previous_start, previous_length).clone()
                    for previous_start, previous_length in written_chunks
                ]
                + [output]
                for result, result_dim, output in zip(results, result_dims, current_outputs, strict=True)
            ]
            return _concatenate(collected, chunks)

        for result, result_dim, output in zip(results, result_dims, current_outputs, strict=True):
            result.narrow(result_dim, start, length).copy_(output)
        written_chunks.append((start, length))
        # Do not keep the just-copied source alive while Python evaluates the
        # next operator call. At long sequence lengths one row block is large.
        del output, current_outputs, current

    return _pack(results)
