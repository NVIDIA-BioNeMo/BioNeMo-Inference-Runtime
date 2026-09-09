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
"""OpenFold3 feature factory: generators and collators for the feature stage."""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from bionemo_ir.pipeline.base import (
    FeatureCollatorSpec,
    FeatureFactoryBase,
    FeatureGeneratorSpec,
    default_context_and_feature_merger,
)

from .feature_collators import OpenFold3FinalFeatureCollator
from .feature_generators import (
    ConformerFeatureGenerator,
    MsaFeatureGenerator,
    StructureFeatureGenerator,
    TemplateFeatureGenerator,
)


def pre_init(context: dict[str, Any]) -> dict[str, Any]:
    """Seed Python, NumPy, and Torch RNGs from context['random_seed'].

    Must run *before* OpenFold3ContextGenerator / RDKit ETKDGv3 (tokenizer
    stage). The processor wires this hook into both the tokenizer stage and
    the feature stage so ETKDG and centre_random_augmentation stay aligned.
    """
    seed = context.get("random_seed", 0)
    if seed is None:
        seed = 0
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return context


class FeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init

    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(
            name="structure",
            functor=StructureFeatureGenerator,
            kwargs={},
        ),
        FeatureGeneratorSpec(
            name="conformer",
            functor=ConformerFeatureGenerator,
            kwargs={},
        ),
        FeatureGeneratorSpec(
            name="msa",
            functor=MsaFeatureGenerator,
            kwargs={},
        ),
        FeatureGeneratorSpec(
            name="template",
            functor=TemplateFeatureGenerator,
            kwargs={},
        ),
    ]

    features_merger_func: Callable = default_context_and_feature_merger

    feature_collator_specs: list[FeatureCollatorSpec] = [
        FeatureCollatorSpec(
            name="openfold3_final_feature_collator",
            functor=OpenFold3FinalFeatureCollator,
            kwargs={},
        ),
    ]
