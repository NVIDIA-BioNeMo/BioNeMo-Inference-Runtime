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
"""Process-cached checkpoint loads for the reference implementations.

Every ``Ref*.load_weights`` classmethod under ``test_utils`` falls back to
loading the whole checkpoint when its caller passes no ``state_dict``, and most
of them build sub-modules without handing their own ``state_dict`` down. One
``RefDiffusionModule.load_weights()`` therefore loaded boltz-2 **74 times**,
~4.2s apiece — 311s of the ~300s each ``test_diffusion_module``
parameterisation spent in CI, which is why its runtime was indifferent to
dtype, multiplicity and attention backend.

Each load re-runs ``verify_boltz_checkpoint_md5`` over the 1.94 GiB file and
deserializes 5102 tensors, for bytes that cannot change within a process. This
wrapper caches that, so the redundant loads collapse into dictionary lookups.

``maxsize=1`` bounds the cost at one checkpoint per pytest worker (~2 GiB for
boltz-2; phase 1 runs 8 of them). The redundant loads are always a burst of the
same checkpoint, so a single slot captures effectively all of the win; a larger
cache would multiply the resident weights across workers for the far rarer case
of a worker alternating between families. ``test_protenix_diffusion_module.py``
caches its own loads the same way.

CALLERS MUST TREAT THE RESULT AS READ-ONLY — it is now shared. Today they do:
every reference loader uses it as the source of ``layer.weight.data.copy_()``
and never as a destination, and none wraps its tensors in an ``nn.Parameter``
(which would alias the storage into a module).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from bionemo_ir.hubs import load_weights as _load_weights_uncached


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
