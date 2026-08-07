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
from typing import Any

from tensorrt_bionemo.pipeline.base import ContextGeneratorBase, TransformBase, dict_context_merger, numpy_to_dict
from tensorrt_bionemo.pipeline.stages.base import StatefulStage, StatefulStageUDF


class TokenizerUDF(StatefulStageUDF):
    def __init__(
        self,
        compute_by_rows: bool,
        drop_keys: list[str],
        expected_input_keys: list[str],
        update_row: bool,
        context_generators: dict[str, ContextGeneratorBase],
        context_merger_func: Callable | None = dict_context_merger,
        transform_funcs: list[TransformBase] | None = None,
        pre_init: Callable | None = None,
        init_context: dict[str, Any] | None = None,
    ):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys, update_row)
        self.context_generators = context_generators
        self.context_merger_func = context_merger_func
        self.transform_funcs = transform_funcs or []
        self.pre_init = pre_init
        self.init_context = init_context

    async def udf_for_item(self, row: dict[str, Any]) -> dict[str, Any]:
        # Seed before context generation so RDKit ETKDG (OpenFold3) sees the
        # same RNG state as FeatureFactory.pre_init documents.
        if self.pre_init is not None:
            from copy import deepcopy

            ctx = deepcopy(self.init_context) if self.init_context is not None else {}
            self.pre_init(context=ctx)

        context_dict = {}
        for name, generator in self.context_generators.items():
            required_kwargs = generator.required_kwargs
            if required_kwargs:
                required_kwargs_dict = {k: row[k] for k in required_kwargs}
                required_kwargs_dict = numpy_to_dict(required_kwargs_dict)
                context_dict[name] = generator(**required_kwargs_dict)
            else:
                context_dict[name] = generator()
        context_dict = self.context_merger_func(context_dict)
        for transform_func in self.transform_funcs:
            if transform_func.is_enabled():
                context_dict = transform_func(context_dict)

        return context_dict


class TokenizerStage(StatefulStage):
    """
    A stage that tokenizes the input.
    """

    fn: type[StatefulStageUDF] = TokenizerUDF

    def get_required_input_keys(self) -> dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {"parsed": "A parsed record of the input. See tensorrt_bionemo.data.schemas.InputParsed for details."}
