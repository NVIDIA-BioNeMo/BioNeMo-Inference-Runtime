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

"""Support for the test suite, deliberately not shipped.

`pyproject.toml` leaves this package out of wheel discovery; the suite reaches
it only because pytest puts the checkout on `sys.path`. Anything a caller
outside the suite would want belongs in `hubs.__all__` instead.
"""

from __future__ import annotations

from ..hf import HF_CHECKPOINTS
from ..local import LOCAL_CHECKPOINTS, _resolve_local_checkpoint

__all__ = ["checkpoint_available"]


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
