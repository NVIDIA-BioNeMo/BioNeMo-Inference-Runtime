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
"""OpenFold3 feature collators.

Produces the final feature set: adds batch dimension and optionally
filters feature keys.
"""

from typing import Any

import torch

from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import FeatureCollatorBase


class OpenFold3FinalFeatureCollator(FeatureCollatorBase):
    """Produces the final feature set with a leading batch dimension.

    Adds a leading batch dimension to every tensor value (matching the
    model's expected input convention where all features are ``(1, ...)``).
    """

    def __init__(
        self,
        config: BaseConfig | None = None,
        include_feats: list[str] | None = None,
        exclude_feats: list[str] | None = None,
        add_batch_dim: bool = True,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self.include_feats = include_feats
        self.exclude_feats = exclude_feats or [
            "structure",
            "msa_per_chain",
            "paired_msa_per_chain",
            "chain_sequences",
        ]
        self.add_batch_dim = add_batch_dim

    def __call__(
        self,
        features: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Work on a shallow copy so we don't mutate the caller's dict.
        out = dict(features)

        if self.exclude_feats is not None:
            for k in self.exclude_feats:
                out.pop(k, None)

        if self.include_feats is not None:
            out = {k: v for k, v in out.items() if k in self.include_feats}

        if self.add_batch_dim:
            out = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in out.items()}

        return out
