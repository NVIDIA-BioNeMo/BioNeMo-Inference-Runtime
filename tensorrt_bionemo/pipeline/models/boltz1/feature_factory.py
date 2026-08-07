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

"""Boltz1 feature factory: generators and collators for the feature stage."""

from collections.abc import Callable
from typing import Any

from tensorrt_bionemo.pipeline.base import (
    FeatureCollatorSpec,
    FeatureFactoryBase,
    FeatureGeneratorSpec,
    default_context_and_feature_merger,
)

from .feature_collators import Boltz1FinalFeatureCollator
from .feature_generators import (
    Boltz1AtomFeatureGenerator,
    Boltz1ChainConstraintFeatureGenerator,
    Boltz1MsaFeatureGenerator,
    Boltz1ResidueConstraintFeatureGenerator,
    Boltz1TokenFeatureGenerator,
)


def pre_init(context: dict[str, Any]) -> dict[str, Any]:
    return context


class FeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(name="token", functor=Boltz1TokenFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(name="atom", functor=Boltz1AtomFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(name="msa", functor=Boltz1MsaFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(
            name="residue_constraint",
            functor=Boltz1ResidueConstraintFeatureGenerator,
            kwargs={},
        ),
        FeatureGeneratorSpec(
            name="chain_constraint",
            functor=Boltz1ChainConstraintFeatureGenerator,
            kwargs={},
        ),
    ]
    features_merger_func: Callable = default_context_and_feature_merger
    feature_collator_specs: list[FeatureCollatorSpec] = [
        FeatureCollatorSpec(
            name="boltz1_final_feature_collator",
            functor=Boltz1FinalFeatureCollator,
            kwargs={},
        ),
    ]
