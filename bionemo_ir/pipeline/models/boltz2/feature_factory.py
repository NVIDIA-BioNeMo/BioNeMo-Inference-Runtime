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
"""Boltz2 feature factory: generators and collators for the feature stage.

Metadata (e.g. ccd_path, mol_dir) is not used here; it is only used by the
Tokenizer stage (see Boltz2ContextGenerator in feature_context.py).
"""

from collections.abc import Callable
from typing import Any

# isort: off
from bionemo_ir.pipeline.base import (
    FeatureCollatorSpec,
    FeatureFactoryBase,
    FeatureGeneratorSpec,
    default_context_and_feature_merger,
)

from .feature_collators import Boltz2FinalFeatureCollator
from .feature_generators import (
    Boltz2AtomFeatureGenerator,
    Boltz2ChainConstraintFeatureGenerator,
    Boltz2ContactConstraintFeatureGenerator,
    Boltz2EnsembleFeatureGenerator,
    Boltz2MsaFeatureGenerator,
    Boltz2ResidueConstraintFeatureGenerator,
    Boltz2TemplateFeatureGenerator,
    Boltz2TokenFeatureGenerator,
)
# isort: on


def pre_init(context: dict[str, Any]) -> dict[str, Any]:
    """No-op pre-init; all seeding is handled inside the ContextGenerator."""
    return context


class FeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(name="token", functor=Boltz2TokenFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(name="ensemble", functor=Boltz2EnsembleFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(name="atom", functor=Boltz2AtomFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(name="msa", functor=Boltz2MsaFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(name="template", functor=Boltz2TemplateFeatureGenerator, kwargs={}),
        FeatureGeneratorSpec(
            name="residue_constraint",
            functor=Boltz2ResidueConstraintFeatureGenerator,
            kwargs={},
        ),
        FeatureGeneratorSpec(
            name="chain_constraint",
            functor=Boltz2ChainConstraintFeatureGenerator,
            kwargs={},
        ),
        FeatureGeneratorSpec(
            name="contact_constraint",
            functor=Boltz2ContactConstraintFeatureGenerator,
            kwargs={},
        ),
    ]
    features_merger_func: Callable = default_context_and_feature_merger
    feature_collator_specs: list[FeatureCollatorSpec] = [
        FeatureCollatorSpec(
            name="boltz2_final_feature_collator",
            functor=Boltz2FinalFeatureCollator,
            kwargs={},
        ),
    ]
