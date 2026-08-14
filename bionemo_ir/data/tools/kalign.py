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
"""Kalign MSA wrapper (model-agnostic).

Thin wrapper over ``kalign-python`` (``kalign.align``) — the same library OSS
``run_kalign`` uses. Kept under ``data/tools`` for reuse across models.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=512)
def _run_kalign_cached(sequences: tuple[str, ...]) -> tuple[str, ...]:
    import kalign  # pip: kalign-python (same lib OSS run_kalign imports)

    return tuple(kalign.align(list(sequences)))


def run_kalign(sequences: list[str]) -> list[str]:
    """Align sequences with kalign, returning aligned (gapped) strings.

    First sequence is the query; the rest are templates. Mirrors the
    upstream OpenFold3 ``run_kalign``
    (``openfold3/core/data/tools/kalign.py``) including the ``lru_cache``.
    """
    return list(_run_kalign_cached(tuple(sequences)))
