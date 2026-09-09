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
"""Shared contracts for Torch-native iterative sampling."""

from __future__ import annotations

import contextlib
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

from bionemo_ir._torch.utils import safe_generator

StateT = TypeVar("StateT")
PlanT = TypeVar("PlanT")
ScheduleT = TypeVar("ScheduleT")
StepT = TypeVar("StepT")
ModelInputT = TypeVar("ModelInputT")
TimeT = TypeVar("TimeT")
PredictionT = TypeVar("PredictionT")


class SamplingScheduleConfig[ScheduleT](ABC):
    """Build a resolved schedule for a number of model evaluations."""

    @abstractmethod
    def build(
        self,
        num_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ScheduleT:
        """Create a device-local schedule containing ``num_steps`` intervals.

        ``device`` defaults to CPU when omitted.
        """


class SamplingIntegratorConfig(ABC):
    """Validated configuration for one mathematical integration method."""

    @abstractmethod
    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is invalid."""


class SamplingRolloutPlan[ScheduleT](ABC):
    """Resolved schedule and execution placement for one rollout."""

    schedule: ScheduleT
    device: torch.device
    dtype: torch.dtype

    @property
    @abstractmethod
    def num_steps(self) -> int:
        """Number of integration intervals and model evaluations."""


@dataclass
class SamplingContext:
    """Mutable execution state owned by one sampling rollout."""

    generator: torch.Generator
    seed: int
    step_index: int = 0
    is_cancelled: Callable[[], bool] | None = None

    @classmethod
    def create(cls, device: torch.device, seed: int | None = None) -> SamplingContext:
        """Create a device-local generator without using Torch's global RNG."""
        seed = secrets.randbits(63) if seed is None else seed
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return cls(generator=generator, seed=seed)

    @classmethod
    @contextlib.contextmanager
    def graph_safe(
        cls,
        device: torch.device,
        seed: int | None = None,
    ) -> Iterator[SamplingContext]:
        """Create a graph-safe context for one sampling rollout.

        Without an explicit seed, the private generator clones the default
        device RNG and commits its advanced state on successful exit. This
        preserves the legacy RNG stream while keeping eager sampling draws off
        the default CUDA generator registered by CUDA Graph capture.

        An explicit seed already defines an independent deterministic stream,
        so that path leaves the default generator untouched.
        """
        if seed is not None:
            yield cls.create(device, seed)
            return

        with safe_generator(device) as generator:
            yield cls(generator=generator, seed=generator.initial_seed())


@dataclass(frozen=True)
class SamplingResult[StateT]:
    """Result of a completed iterative rollout."""

    final_state: StateT
    seed: int
    num_steps: int


PredictFn = Callable[[ModelInputT, TimeT], PredictionT]


class StepIntegrator(Protocol[PlanT, StateT, ModelInputT, TimeT, PredictionT]):
    """Mathematical state transition driven by :class:`GenerativeRunner`."""

    def num_steps(self, plan: PlanT) -> int:
        """Return the number of model evaluations in ``plan``."""

    def initialize(self, plan: PlanT, context: SamplingContext) -> StateT:
        """Sample or otherwise construct the initial state."""

    def step(
        self,
        step_index: int,
        state: StateT,
        plan: PlanT,
        predict: PredictFn[ModelInputT, TimeT, PredictionT],
        context: SamplingContext,
    ) -> StateT:
        """Advance ``state`` by one integration interval."""
