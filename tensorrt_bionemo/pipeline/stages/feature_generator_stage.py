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

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch

from tensorrt_bionemo.pipeline.base import (FeatureCollatorBase,
                                            FeatureGeneratorBase,
                                            dict_context_merger)
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class FeatureGeneratorUDF(StatefulStageUDF):

    def __init__(
            self,
            compute_by_rows: bool,
            drop_keys: List[str],
            expected_input_keys: List[str],
            update_row: bool,
            feature_generators: list[FeatureGeneratorBase],
            features_merger_func: Optional[Callable] = dict_context_merger,
            feature_collators: Optional[list[FeatureCollatorBase]] = None,
            pre_init: Optional[Callable] = None,
            init_context: Optional[dict[str, Any]] = None):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys,
                         update_row)
        self.feature_generators = feature_generators
        self.features_merger_func = features_merger_func
        self.feature_collators = feature_collators or []
        self.pre_init = pre_init
        self.init_context = init_context

    def _extract_tensors(self, row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        row_with_tensors = {}
        for k, v in row.items():
            if isinstance(v, torch.Tensor):
                row_with_tensors[k] = v
            elif isinstance(v, np.ndarray):
                row_with_tensors[k] = torch.from_numpy(v)
            elif isinstance(v, np.generic):
                row_with_tensors[k] = torch.tensor(v)
        return row_with_tensors

    async def udf_for_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row_with_tensors = self._extract_tensors(row)
        # Per-row copy of the initial context; default to an empty dict when None.
        context = deepcopy(
            self.init_context) if self.init_context is not None else {}
        features_dict = {}

        if self.pre_init is not None:
            context = self.pre_init(context=context)

        with torch.no_grad():
            for generator in self.feature_generators:
                if generator.is_enabled():
                    if generator.name in features_dict:
                        raise ValueError(
                            f"Feature generator {generator.name} is already in the features dictionary."
                        )
                    features_dict[generator.name] = generator(
                        row_with_tensors, context)

            merged_feats = self.features_merger_func(contexts=row_with_tensors,
                                                     features=features_dict)
            for collator in self.feature_collators:
                if collator.is_enabled():
                    merged_feats = collator(merged_feats, context)

        return merged_feats


class FeatureGeneratorStage(StatefulStage):
    """
    A stage that tokenizes the input.
    """

    fn: Type[StatefulStageUDF] = FeatureGeneratorUDF
    update_row: bool = False
