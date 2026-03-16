# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boltz1 ContextGenerator: reuses Boltz2 structure building and tokenization.

Boltz1 does not use mol_dir/templates; it only needs CCD and optionally MSA.
"""

from __future__ import annotations

from typing import Any, Optional

from tensorrt_bionemo.pipeline.models.boltz2.feature_context import \
    Boltz2ContextGenerator


class Boltz1ContextGenerator(Boltz2ContextGenerator):
    """Boltz1 context generator.

    Identical to Boltz2: structure + CCD + tokenization + MSA.
    Boltz1 has no template/extra-mol features, but the same context dict works
    because the feature generators simply don't request those fields.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        ccd_path=None,
        mol_dir=None,
        **kwargs: Any,
    ):
        super().__init__(config=config,
                         metadata=metadata,
                         ccd_path=ccd_path,
                         mol_dir=mol_dir,
                         **kwargs)
