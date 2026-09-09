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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch.nn as nn

from bionemo_ir.hubs import FoldingSupportMatrix as SupMat
from bionemo_ir.logger import logger

if TYPE_CHECKING:
    from bionemo_ir.pipeline.base import FeatureFactoryBase, PostProcessorBase, TokenizerBase


class ModelComponentsFactory(ABC):
    @classmethod
    @abstractmethod
    def get_model_class(cls) -> type[nn.Module]:
        pass

    @classmethod
    @abstractmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        pass

    @classmethod
    @abstractmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        pass

    @classmethod
    @abstractmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        pass

    @classmethod
    def get_default_runtime_args(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return []


class ModelRegistry:
    _factories: ClassVar[dict[str, type[ModelComponentsFactory]]] = {}

    @classmethod
    def get_models(cls) -> list[str]:
        return sorted(cls._factories)

    @classmethod
    def register(cls, model_name: str, factory: type[ModelComponentsFactory]):
        if model_name in cls._factories:
            logger.warning(f"Factory for model {model_name} already registered, overriding")
        cls._factories[model_name] = factory

    @classmethod
    def register_factory(cls, factory: type[ModelComponentsFactory]):
        for model_name in factory.get_supported_model_names():
            cls.register(model_name, factory)

    @classmethod
    def get_factory(cls, model_name: str) -> type[ModelComponentsFactory]:
        if model_name not in cls._factories:
            raise ValueError(f"Model {model_name} not registered. Available models: {list(cls._factories.keys())}")
        return cls._factories[model_name]

    @classmethod
    def get_model_class(cls, model_name: str) -> type[nn.Module]:
        return cls.get_factory(model_name).get_model_class()

    @classmethod
    def get_tokenizer(cls, model_name: str) -> "TokenizerBase":
        return cls.get_factory(model_name).get_tokenizer()

    @classmethod
    def get_feature_factory(cls, model_name: str) -> "FeatureFactoryBase":
        return cls.get_factory(model_name).get_feature_factory()

    @classmethod
    def get_postprocessor(cls, model_name: str) -> type["PostProcessorBase"]:
        return cls.get_factory(model_name).get_postprocessor()

    @classmethod
    def get_default_runtime_args(cls, model_name: str) -> dict[str, Any]:
        return cls.get_factory(model_name).get_default_runtime_args()

    @classmethod
    def load_metadata(
        cls,
        model_name: str,
        cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        from bionemo_ir.hubs.metadata import load_metadata

        return load_metadata(model_name, cache_dir)


class OpenFold2Factory(ModelComponentsFactory):
    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.openfold2 import OpenFold2

        return OpenFold2

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from bionemo_ir.pipeline.models.openfold2.tokenizer import Tokenizer

        return Tokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from bionemo_ir.pipeline.models.openfold2.feature_factory import FeatureFactory

        return FeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        from bionemo_ir.pipeline.models.openfold2.postprocessor import PostProcessor

        return PostProcessor

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [
            SupMat.OpenFold2_FT2,
            SupMat.OpenFold2_FT3,
            SupMat.OpenFold2_FT4,
            SupMat.OpenFold2_FT5,
            SupMat.OpenFold2_NoTempl1,
            SupMat.OpenFold2_NoTempl2,
            SupMat.OpenFold2_NoTempl_PTM1,
            SupMat.OpenFold2_PTM1,
            SupMat.OpenFold2_PTM2,
            SupMat.AlphaFold2_1,
            SupMat.AlphaFold2_2,
            SupMat.AlphaFold2_3,
            SupMat.AlphaFold2_4,
            SupMat.AlphaFold2_5,
        ]


class OpenFold2MultimerFactory(ModelComponentsFactory):
    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.openfold2 import OpenFold2

        return OpenFold2

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from bionemo_ir.pipeline.models.openfold2.tokenizer import MultimerTokenizer

        return MultimerTokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from bionemo_ir.pipeline.models.openfold2.feature_factory import MultimerFeatureFactory

        return MultimerFeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        from bionemo_ir.pipeline.models.openfold2.postprocessor import PostProcessor

        return PostProcessor

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [
            SupMat.AlphaFold2_Multimer_1,
            SupMat.AlphaFold2_Multimer_2,
            SupMat.AlphaFold2_Multimer_3,
            SupMat.AlphaFold2_Multimer_4,
            SupMat.AlphaFold2_Multimer_5,
        ]


class Boltz1Factory(ModelComponentsFactory):
    @classmethod
    def get_default_runtime_args(cls) -> dict[str, Any]:
        return {
            "recycling_steps": 3,
            "num_sampling_steps": 200,
            "diffusion_samples": 1,
        }

    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.boltz1 import Boltz1

        return Boltz1

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from bionemo_ir.pipeline.models.boltz1.tokenizer import Tokenizer

        return Tokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from bionemo_ir.pipeline.models.boltz1.feature_factory import FeatureFactory

        return FeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        from bionemo_ir.pipeline.models.boltz2.postprocessor import PostProcessor

        return PostProcessor

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.Boltz1]


class Boltz2Factory(ModelComponentsFactory):
    @classmethod
    def get_default_runtime_args(cls) -> dict[str, Any]:
        return {
            "recycling_steps": 3,
            "num_sampling_steps": 200,
            "diffusion_samples": 1,
        }

    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.boltz2 import Boltz2

        return Boltz2

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from bionemo_ir.pipeline.models.boltz2.tokenizer import Tokenizer

        return Tokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from bionemo_ir.pipeline.models.boltz2.feature_factory import FeatureFactory

        return FeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        from bionemo_ir.pipeline.models.boltz2.postprocessor import PostProcessor

        return PostProcessor

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.Boltz2]


class Boltz2AffinityFactory(ModelComponentsFactory):
    @classmethod
    def get_default_runtime_args(cls) -> dict[str, Any]:
        return {
            "recycling_steps": 3,
            "num_sampling_steps": 200,
            "diffusion_samples": 1,
        }

    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.boltz2 import Boltz2Affinity

        return Boltz2Affinity

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        raise NotImplementedError("Boltz2Affinity tokenizer not implemented in pipeline")

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        raise NotImplementedError("Boltz2Affinity feature factory not implemented in pipeline")

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        raise NotImplementedError("Boltz2Affinity postprocessor not implemented in pipeline")

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.Boltz2Affinity]


class OpenFold3Factory(ModelComponentsFactory):
    @classmethod
    def get_default_runtime_args(cls) -> dict[str, Any]:
        # Boltz-style kwarg names so the generic ``FoldingEngine`` can forward
        # the same ``runtime_args`` dict to either model. Mapping to OF3
        # internals (see ``OpenFold3.forward``):
        #   recycling_steps    → num_cycles = recycling_steps + 1
        #   num_sampling_steps → no_rollout_steps   (diffusion rollout length)
        #   diffusion_samples  → no_rollout_samples (parallel rollout samples)
        return {
            "recycling_steps": 3,
            "num_sampling_steps": 200,
            "diffusion_samples": 1,
        }

    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.openfold3 import OpenFold3

        return OpenFold3

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from bionemo_ir.pipeline.models.openfold3.tokenizer import Tokenizer

        return Tokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from bionemo_ir.pipeline.models.openfold3.feature_factory import FeatureFactory

        return FeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        from bionemo_ir.pipeline.models.openfold3.postprocessor import PostProcessor

        return PostProcessor

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.OpenFold3]


class ProtenixFactory(ModelComponentsFactory):
    @classmethod
    def get_model_class(cls) -> type[nn.Module]:
        from bionemo_ir.models.protenix import Protenix

        return Protenix

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        raise NotImplementedError("Protenix tokenizer not implemented in pipeline")

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        raise NotImplementedError("Protenix feature factory not implemented in pipeline")

    @classmethod
    def get_postprocessor(cls) -> type["PostProcessorBase"]:
        raise NotImplementedError("Protenix postprocessor not implemented in pipeline")

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.ProtenixV2]


def register_all_factories():
    factories = [
        OpenFold2Factory,
        OpenFold2MultimerFactory,
        Boltz1Factory,
        Boltz2Factory,
        Boltz2AffinityFactory,
        OpenFold3Factory,
        ProtenixFactory,
    ]
    for factory in factories:
        ModelRegistry.register_factory(factory)

    logger.info(f"Registered {len(factories)} model factories")


def get_model_class(model_name: str) -> type[nn.Module]:
    return ModelRegistry.get_model_class(model_name)


def get_tokenizer(model_name: str) -> "TokenizerBase":
    return ModelRegistry.get_tokenizer(model_name)


def get_feature_factory(model_name: str) -> "FeatureFactoryBase":
    return ModelRegistry.get_feature_factory(model_name)


def get_postprocessor(model_name: str) -> type["PostProcessorBase"]:
    return ModelRegistry.get_postprocessor(model_name)


def get_default_runtime_args(model_name: str) -> dict[str, Any]:
    return ModelRegistry.get_default_runtime_args(model_name)


def load_metadata(
    model_name: str,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    return ModelRegistry.load_metadata(model_name, cache_dir)
