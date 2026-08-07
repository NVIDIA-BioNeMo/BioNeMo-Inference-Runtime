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

"""Boltz1 ContextGenerator: reuses Boltz2 structure building and tokenization.

Boltz-1 shares the Boltz-2 CCD + mols/ assets: the inherited
``Boltz2ContextGenerator.__call__`` always loads per-CCD-residue molecule
pickles for every token (canonical + residue-specific), so ``mol_dir`` is
required here too. Templates remain unused.
"""

from __future__ import annotations

from typing import Any

from tensorrt_bionemo.pipeline.models.boltz2.feature_context import Boltz2ContextGenerator


class Boltz1ContextGenerator(Boltz2ContextGenerator):
    """Boltz1 context generator.

    Identical to Boltz2: structure + CCD + tokenization + MSA.
    Boltz1 has no template/extra-mol features, but the same context dict works
    because the feature generators simply don't request those fields.
    """

    def __init__(
        self,
        config: Any | None = None,
        metadata: dict[str, Any] | None = None,
        ccd_path=None,
        mol_dir=None,
        **kwargs: Any,
    ):
        super().__init__(config=config, metadata=metadata, ccd_path=ccd_path, mol_dir=mol_dir, **kwargs)
