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

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from bionemo_ir.pipeline.base import FeatureCollatorBase, FeatureGeneratorBase, dict_context_merger
from bionemo_ir.pipeline.stages.base import StatefulStage, StatefulStageUDF


class FeatureGeneratorUDF(StatefulStageUDF):
    def __init__(
        self,
        compute_by_rows: bool,
        drop_keys: list[str],
        expected_input_keys: list[str],
        update_row: bool,
        feature_generators: list[FeatureGeneratorBase],
        features_merger_func: Callable | None = dict_context_merger,
        feature_collators: list[FeatureCollatorBase] | None = None,
        pre_init: Callable | None = None,
        init_context: dict[str, Any] | None = None,
    ):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys, update_row)
        self.feature_generators = feature_generators
        self.features_merger_func = features_merger_func
        self.feature_collators = feature_collators or []
        self.pre_init = pre_init
        self.init_context = init_context

    def _extract_tensors(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        row_with_tensors = {}
        for k, v in row.items():
            if isinstance(v, torch.Tensor):
                row_with_tensors[k] = v
            elif isinstance(v, np.ndarray):
                if v.dtype == object:
                    continue
                arr = np.asarray(v, order="C")
                if not arr.flags.writeable:
                    arr = arr.copy()
                row_with_tensors[k] = torch.from_numpy(arr)
            elif isinstance(v, np.generic):
                # Use .item() to avoid DeprecationWarning for np.bool_ etc. as index
                row_with_tensors[k] = torch.tensor(v.item())
        return row_with_tensors

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        row_with_tensors = self._extract_tensors(row)
        # Per-row copy of the initial context; default to an empty dict when None.
        context = deepcopy(self.init_context) if self.init_context is not None else {}
        # Expose full row to generators (e.g. Boltz2 needs structure, tokens, molecules, MSA).
        context["_row"] = row
        features_dict = {}

        if self.pre_init is not None:
            context = self.pre_init(context=context)

        with torch.no_grad():
            merged_feats = row_with_tensors
            for generator in self.feature_generators:
                if generator.is_enabled():
                    if generator.name in features_dict:
                        raise ValueError(
                            f"Feature generator '{generator.name}' conflicts "
                            f"with a previously generated feature in "
                            f"features_dict."
                        )
                    features_dict[generator.name] = generator(merged_feats, context)
                    merged_feats = self.features_merger_func(
                        contexts=merged_feats,
                        features={generator.name: features_dict[generator.name]},
                    )
            for collator in self.feature_collators:
                if collator.is_enabled():
                    merged_feats = collator(merged_feats, context)

        return merged_feats


class FeatureGeneratorStage(StatefulStage):
    """
    A stage that tokenizes the input.
    """

    fn: type[StatefulStageUDF] = FeatureGeneratorUDF
    update_row: bool = False
