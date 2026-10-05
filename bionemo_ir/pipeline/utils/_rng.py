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
"""Request-local randomness for CPU feature preparation."""

import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType

import numpy as np
import torch


class _RequestRNG:
    def __init__(self, seed: int, torch_seed: int | None = None) -> None:
        self.python = random.Random(seed)
        self.numpy = np.random.RandomState(seed)
        self.torch = torch.Generator(device="cpu")
        self.torch.manual_seed(seed if torch_seed is None else torch_seed)


_RNG: ContextVar[_RequestRNG | None] = ContextVar("bioir_feature_rng", default=None)


@contextmanager
def _feature_rng(rng: _RequestRNG | None) -> Iterator[None]:
    token = _RNG.set(rng)
    try:
        yield
    finally:
        _RNG.reset(token)


def _torch_generator() -> torch.Generator | None:
    rng = _RNG.get()
    return None if rng is None else rng.torch


def _python_rng() -> random.Random | ModuleType:
    rng = _RNG.get()
    return random if rng is None else rng.python
