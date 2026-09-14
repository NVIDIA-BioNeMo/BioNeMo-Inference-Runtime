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
"""Bounded, signature-aware cache for cuDNN operation-graph plans."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Hashable, Sequence

import cudnn
import torch
from lru import LRU

type TensorLayout = tuple[torch.dtype, torch.Size, tuple[int, ...]]
type GraphSignature = tuple[torch.device, tuple[TensorLayout, ...]]

DEFAULT_MAX_CACHED_GRAPHS = 32


def _graph_signature(tensors: Sequence[torch.Tensor]) -> GraphSignature:
    if not tensors:
        raise ValueError("a cuDNN graph signature requires at least one tensor")
    device = tensors[0].device
    layouts: list[TensorLayout] = []
    for tensor in tensors:
        if tensor.device != device:
            raise ValueError(f"all cuDNN graph inputs must use {device}, got {tensor.device}")
        layouts.append((tensor.dtype, tensor.shape, tensor.stride()))
    return device, tuple(layouts)


class CudnnGraphCache[Plan]:
    """Thread-safe LRU cache keyed by complete tensor layout signatures.

    Args:
        name: Human-readable operation-graph name used in fallback warnings.
        max_size: Maximum number of successful and unsupported signatures to
            remember independently.
    """

    def __init__(self, name: str, max_size: int = DEFAULT_MAX_CACHED_GRAPHS) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.name = name
        self.max_size = max_size
        self._plans = LRU(size=max_size)
        self._unsupported = LRU(size=max_size)
        self._lock = threading.Lock()

    def get_or_create(
        self,
        tensors: Sequence[torch.Tensor],
        factory: Callable[[], Plan],
    ) -> Plan | None:
        """Return a cached plan, build it once, or remember unsupported layouts.

        Args:
            tensors: Graph inputs whose device, dtype, shape, and stride define
                the plan signature.
            factory: Callable that builds a plan for the supplied signature.

        Returns:
            The cached or newly built plan, or ``None`` when cuDNN rejects the
            signature.
        """
        return self.get_or_create_signature(_graph_signature(tensors), factory)

    def get_or_create_signature(
        self,
        signature: Hashable,
        factory: Callable[[], Plan],
    ) -> Plan | None:
        """Return a plan cached under an explicit graph signature.

        This entry point supports dynamic operation graphs whose cache keys
        contain only plan-invariant dimensions and layouts.

        Args:
            signature: Hashable execution-plan signature.
            factory: Callable that builds a plan for the supplied signature.

        Returns:
            The cached or newly built plan, or ``None`` when cuDNN rejects the
            signature.
        """
        with self._lock:
            if signature in self._unsupported:
                self._unsupported.get(signature)
                return None
            plan = self._plans.get(signature)
            if plan is not None:
                return plan
            try:
                plan = factory()
            except cudnn.cudnnGraphNotSupportedError as error:
                self._unsupported[signature] = True
                warnings.warn(
                    f"cuDNN does not support {self.name} for tensor signature {signature}; "
                    f"using the caller's fallback: {error}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return None
            self._plans[signature] = plan
            return plan

    def clear(self) -> None:
        """Discard cached plans and unsupported signatures."""
        with self._lock:
            self._plans.clear()
            self._unsupported.clear()

    def __len__(self) -> int:
        """Return the number of successful plans currently cached."""
        with self._lock:
            return len(self._plans)
