# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Dimension Specification System ──────────────────────────────────────


class DimKind(Enum):
    """Classification of a tensor dimension for compilation."""
    STATIC = auto()
    DYNAMIC = auto()
    BATCH = auto()


@dataclass(frozen=True)
class DimSpec:
    """Describes a single named dimension for shape contracts.

    Used by both the torch.compile path (kind / min_val / max_val /
    multiple_of) and the TRT engine-build path (size).
    """
    name: str
    kind: DimKind
    min_val: Optional[int] = None
    max_val: Optional[int] = None
    multiple_of: Optional[int] = None
    size: Optional[int] = None

    def to_export_dim(self) -> Optional["torch.export.Dim"]:
        """Convert to a torch.export.Dim for AOT export."""
        if self.kind == DimKind.STATIC:
            return None
        kwargs: dict[str, int] = {}
        if self.min_val is not None:
            kwargs["min"] = self.min_val
        if self.max_val is not None:
            kwargs["max"] = self.max_val
        return torch.export.Dim(self.name, **kwargs)


@dataclass(frozen=True)
class TensorSpec:
    """Shape contract for a single tensor argument."""
    name: str
    dims: tuple[DimSpec, ...]
    dtype: torch.dtype = torch.float32

    @property
    def dynamic_dim_indices(self) -> list[int]:
        return [
            i for i, d in enumerate(self.dims) if d.kind == DimKind.DYNAMIC
        ]

    @property
    def batch_dim_indices(self) -> list[int]:
        return [i for i, d in enumerate(self.dims) if d.kind == DimKind.BATCH]


class DimRegistry:
    """Global registry ensuring dimension semantics are consistent across
    modules, with per-model dimension lookup."""

    def __init__(self):
        self._dims: dict[str, DimSpec] = {}
        self._model_dims: dict[str, list[DimSpec]] = {}

    def register(self, spec: DimSpec) -> DimSpec:
        if spec.name in self._dims:
            existing = self._dims[spec.name]
            if existing != spec:
                raise ValueError(
                    f"Dim '{spec.name}' already registered with different spec: "
                    f"{existing} vs {spec}")
            return existing
        self._dims[spec.name] = spec
        return spec

    def register_model(self, model_name: str, dims: list[DimSpec]):
        """Associate a set of dimensions with a model name."""
        self._model_dims[model_name] = list(dims)

    def for_model(self, model_name: str) -> list[DimSpec]:
        """Return dimensions used by *model_name*."""
        if model_name not in self._model_dims:
            raise ValueError(f"No dims registered for model '{model_name}'. "
                             f"Available: {list(self._model_dims.keys())}")
        return list(self._model_dims[model_name])

    def get(self, name: str) -> DimSpec:
        return self._dims[name]

    def __contains__(self, name: str) -> bool:
        return name in self._dims


DIMS = DimRegistry()

# ── Dimension vocabulary (model-agnostic) ────────────────────────────────
# Every name is generic — no model names embedded.  Use
# ``DIMS.for_model(name)`` to discover which dims a model needs.

# Batch-like (never padded)
BATCH = DIMS.register(DimSpec("batch", DimKind.BATCH))
MULTIPLICITY = DIMS.register(DimSpec("multiplicity", DimKind.BATCH))

# Dynamic (padded / bucketed)
N_RES = DIMS.register(
    DimSpec("N_res", DimKind.DYNAMIC, min_val=1, max_val=4096, multiple_of=8))
N_SEQ = DIMS.register(
    DimSpec("N_seq", DimKind.DYNAMIC, min_val=1, max_val=2048))
N_SEQ_DYN = DIMS.register(
    DimSpec("N_seq_dyn", DimKind.DYNAMIC, min_val=1, max_val=16384))
N_MSA = DIMS.register(
    DimSpec("N_msa", DimKind.DYNAMIC, min_val=1, max_val=4096))

# Static feature channels (never padded)
C_MSA = DIMS.register(DimSpec("C_msa", DimKind.STATIC))
C_PAIR = DIMS.register(DimSpec("C_pair", DimKind.STATIC))
C_SINGLE = DIMS.register(DimSpec("C_single", DimKind.STATIC))
C_ATOM = DIMS.register(DimSpec("C_atom", DimKind.STATIC))
C_COND = DIMS.register(DimSpec("C_cond", DimKind.STATIC))
C_BIAS = DIMS.register(DimSpec("C_bias", DimKind.STATIC))
C_EMB = DIMS.register(DimSpec("C_emb", DimKind.STATIC))
C_MSA_IN = DIMS.register(DimSpec("C_msa_in", DimKind.STATIC))

# ── Per-model dimension sets ─────────────────────────────────────────────
# Follows tensorrt_bionemo.registry model order.

DIMS.register_model("openfold2", [
    BATCH,
    N_RES,
    N_SEQ,
    N_SEQ_DYN,
    C_MSA,
    C_PAIR,
    C_SINGLE,
])
DIMS.register_model("boltz1", [
    BATCH,
    N_RES,
    N_MSA,
    C_PAIR,
    C_SINGLE,
    C_ATOM,
    C_COND,
    C_BIAS,
    C_EMB,
    C_MSA_IN,
    MULTIPLICITY,
])
DIMS.register_model("boltz2", [
    BATCH,
    N_RES,
    N_MSA,
    C_PAIR,
    C_SINGLE,
    C_ATOM,
    C_COND,
    C_BIAS,
    C_EMB,
    C_MSA_IN,
    MULTIPLICITY,
])
DIMS.register_model("openfold3", [
    BATCH,
    N_RES,
    C_PAIR,
    C_SINGLE,
    C_ATOM,
    C_COND,
    C_BIAS,
    MULTIPLICITY,
])
