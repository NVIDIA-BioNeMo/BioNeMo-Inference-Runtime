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
"""Checkpoint availability and process-cached loads for the suite.

Reference loaders reload the whole checkpoint when handed no ``state_dict``,
many times over in a single test. The bytes cannot change within a process, so
``maxsize=1`` turns those repeats into lookups while keeping one checkpoint
resident per pytest worker.

CALLERS MUST TREAT THE RESULT AS READ-ONLY — it is shared. Never mutate it or
wrap its tensors in an ``nn.Parameter``, which aliases the storage.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from bionemo_ir.hubs import load_weights as _load_weights_uncached
from bionemo_ir.hubs.hf import HF_CHECKPOINTS
from bionemo_ir.hubs.local import LOCAL_CHECKPOINTS, _resolve_local_checkpoint


def checkpoint_available(name: str) -> bool:
    """Whether :func:`hubs.load_weights` would find weights for *name* here.

    Asks the loader's own resolver, then the same registry it falls back to, so
    this cannot drift from the behaviour it predicts. Tests gate on it to skip
    rather than fail; AlphaFold2 has no `HF_CHECKPOINTS` entry and needs one.
    """
    if name in LOCAL_CHECKPOINTS:
        path, _ = _resolve_local_checkpoint(name)
        if path:
            return True
    return name in HF_CHECKPOINTS


@functools.lru_cache(maxsize=1)
def load_weights(
    name: str,
    return_raw: bool = False,
    local_files_only: bool = False,
    cache_path: str | Path | None = None,
    repo_id: str | Path | None = None,
    hub: str | None = None,
) -> dict | str | Any:
    """Drop-in for :func:`bionemo_ir.hubs.load_weights`, cached per process."""
    return _load_weights_uncached(
        name,
        return_raw=return_raw,
        local_files_only=local_files_only,
        cache_path=cache_path,
        repo_id=repo_id,
        hub=hub,
    )
