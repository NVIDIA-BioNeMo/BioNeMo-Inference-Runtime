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
"""Boltz2 feature collators.

Per base.py (Feature Collator stage): "Collates feature tensors to produce the
final feature set. Conceptually can add new features, remove features, or
modify features." Boltz2 uses a single collator that produces the final batch
(e.g. optional key filtering / ordering for reproducibility).
"""

from typing import Any

import torch

from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import FeatureCollatorBase


class Boltz2FinalFeatureCollator(FeatureCollatorBase):
    """
    Produces the final feature set from merged context + generator outputs.

    Adds a leading batch dimension to every tensor value (matching the OSS
    Boltz2 dataloader output convention where all features are ``(1, ...)``).
    Can also be configured to restrict keys via include_feats / exclude_feats.
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
        self.exclude_feats = exclude_feats
        self.add_batch_dim = add_batch_dim

    def __call__(
        self,
        features: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if self.include_feats is not None:
            remove = {k for k in features if k not in self.include_feats}
            for k in remove:
                del features[k]
        if self.exclude_feats is not None:
            for k in self.exclude_feats:
                features.pop(k, None)
        if self.add_batch_dim:
            features = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        return features
