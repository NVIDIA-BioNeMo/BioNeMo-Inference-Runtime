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

"""Declare native Protenix feature processing and request seeds."""

from collections.abc import Callable

from bionemo_ir.pipeline.base import (
    FeatureCollatorSpec,
    FeatureFactoryBase,
    FeatureGeneratorSpec,
    default_context_and_feature_merger,
)
from bionemo_ir.pipeline.utils._rng import _initialize_request_rng as pre_init

from .feature_collators import ProtenixFeatureCollator
from .feature_generators import BondFeatureGenerator, ReferenceFeatureGenerator, TokenFeatureGenerator
from .msa import MsaFeatureGenerator
from .templates import TemplateFeatureGenerator


class FeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(name="tokens", functor=TokenFeatureGenerator),
        FeatureGeneratorSpec(name="bonds", functor=BondFeatureGenerator),
        FeatureGeneratorSpec(name="reference", functor=ReferenceFeatureGenerator),
        FeatureGeneratorSpec(name="msa", functor=MsaFeatureGenerator),
        FeatureGeneratorSpec(name="templates", functor=TemplateFeatureGenerator),
    ]
    features_merger_func: Callable = default_context_and_feature_merger
    feature_collator_specs: list[FeatureCollatorSpec] = [
        FeatureCollatorSpec(name="batch", functor=ProtenixFeatureCollator)
    ]
