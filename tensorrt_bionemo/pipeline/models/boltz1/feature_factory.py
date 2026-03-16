# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boltz1 feature factory: generators and collators for the feature stage."""

from typing import Any, Callable

from tensorrt_bionemo.pipeline.base import (FeatureCollatorSpec,
                                            FeatureFactoryBase,
                                            FeatureGeneratorSpec,
                                            default_context_and_feature_merger)

from .feature_collators import Boltz1FinalFeatureCollator
from .feature_generators import (Boltz1AtomFeatureGenerator,
                                 Boltz1ChainConstraintFeatureGenerator,
                                 Boltz1MsaFeatureGenerator,
                                 Boltz1ResidueConstraintFeatureGenerator,
                                 Boltz1TokenFeatureGenerator)


def pre_init(context: dict[str, Any]) -> dict[str, Any]:
    return context


class FeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(name="token",
                             functor=Boltz1TokenFeatureGenerator,
                             kwargs={}),
        FeatureGeneratorSpec(name="atom",
                             functor=Boltz1AtomFeatureGenerator,
                             kwargs={}),
        FeatureGeneratorSpec(name="msa",
                             functor=Boltz1MsaFeatureGenerator,
                             kwargs={}),
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
