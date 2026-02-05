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
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type

import torch.nn as nn

from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.logger import logger

if TYPE_CHECKING:
    from tensorrt_bionemo.pipeline.base import (FeatureFactoryBase,
                                                PostProcessorBase,
                                                TokenizerBase)


class ModelComponentsFactory(ABC):

    @classmethod
    @abstractmethod
    def get_model_class(cls) -> Type[nn.Module]:
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
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        pass

    @classmethod
    @abstractmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        pass

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return []


class ModelRegistry:
    _factories: ClassVar[Dict[str, Type[ModelComponentsFactory]]] = {}

    @classmethod
    def register(cls, model_name: str, factory: Type[ModelComponentsFactory]):
        if model_name in cls._factories:
            logger.warning(
                f"Factory for model {model_name} already registered, overriding"
            )
        cls._factories[model_name] = factory

    @classmethod
    def register_factory(cls, factory: Type[ModelComponentsFactory]):
        for model_name in factory.get_supported_model_names():
            cls.register(model_name, factory)

    @classmethod
    def get_factory(cls, model_name: str) -> Type[ModelComponentsFactory]:
        if model_name not in cls._factories:
            raise ValueError(
                f"Model {model_name} not registered. "
                f"Available models: {list(cls._factories.keys())}")
        return cls._factories[model_name]

    @classmethod
    def get_model_class(cls, model_name: str) -> Type[nn.Module]:
        return cls.get_factory(model_name).get_model_class()

    @classmethod
    def get_tokenizer(cls, model_name: str) -> "TokenizerBase":
        return cls.get_factory(model_name).get_tokenizer()

    @classmethod
    def get_feature_factory(cls, model_name: str) -> "FeatureFactoryBase":
        return cls.get_factory(model_name).get_feature_factory()

    @classmethod
    def get_postprocessor(cls, model_name: str) -> Type["PostProcessorBase"]:
        return cls.get_factory(model_name).get_postprocessor()

    @classmethod
    def get_trt_building_modules(cls, model_name: str) -> Dict[str, Any]:
        return cls.get_factory(model_name).get_trt_building_modules()

    @classmethod
    def get_building_module_class(cls, model_name: str,
                                  module_name: str) -> Any:
        modules = cls.get_trt_building_modules(model_name)
        if module_name not in modules:
            raise ValueError(
                f"Module {module_name} not found for model {model_name}. "
                f"Available modules: {list(modules.keys())}")
        return modules[module_name]


class OpenFold2Factory(ModelComponentsFactory):

    @classmethod
    def get_model_class(cls) -> Type[nn.Module]:
        from tensorrt_bionemo.models.openfold2 import OpenFold2
        return OpenFold2

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from tensorrt_bionemo.pipeline.models.openfold2.tokenizer import \
            Tokenizer
        return Tokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from tensorrt_bionemo.pipeline.models.openfold2.feature_factory import \
            FeatureFactory
        return FeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        from tensorrt_bionemo.pipeline.models.openfold2.postprocessor import \
            PostProcessor
        return PostProcessor

    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        from tensorrt_bionemo._trt.layers.transformers import EvoformerStack
        return {"evoformer": EvoformerStack}

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
            SupMat.AlphaFold2_Multimer_1,
            SupMat.AlphaFold2_Multimer_2,
            SupMat.AlphaFold2_Multimer_3,
            SupMat.AlphaFold2_Multimer_4,
            SupMat.AlphaFold2_Multimer_5,
        ]


class Boltz1Factory(ModelComponentsFactory):

    @classmethod
    def get_model_class(cls) -> Type[nn.Module]:
        from tensorrt_bionemo.models.boltz1 import Boltz1
        return Boltz1

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        raise NotImplementedError(
            "Boltz1 tokenizer not implemented in pipeline")

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        raise NotImplementedError(
            "Boltz1 feature factory not implemented in pipeline")

    @classmethod
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        raise NotImplementedError(
            "Boltz1 postprocessor not implemented in pipeline")

    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        from tensorrt_bionemo._trt.layers.transformers import (
            PairformerModule, TokenTransformer)
        return {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        }

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.Boltz1]


class Boltz2Factory(ModelComponentsFactory):

    @classmethod
    def get_model_class(cls) -> Type[nn.Module]:
        from tensorrt_bionemo.models.boltz2 import Boltz2
        return Boltz2

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        raise NotImplementedError(
            "Boltz2 tokenizer not implemented in pipeline")

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        raise NotImplementedError(
            "Boltz2 feature factory not implemented in pipeline")

    @classmethod
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        raise NotImplementedError(
            "Boltz2 postprocessor not implemented in pipeline")

    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        from tensorrt_bionemo._trt.layers.transformers import (
            PairformerModule, TokenTransformer)
        return {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        }

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.Boltz2]


class Boltz2AffinityFactory(ModelComponentsFactory):

    @classmethod
    def get_model_class(cls) -> Type[nn.Module]:
        from tensorrt_bionemo.models.boltz2 import Boltz2Affinity
        return Boltz2Affinity

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        raise NotImplementedError(
            "Boltz2Affinity tokenizer not implemented in pipeline")

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        raise NotImplementedError(
            "Boltz2Affinity feature factory not implemented in pipeline")

    @classmethod
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        raise NotImplementedError(
            "Boltz2Affinity postprocessor not implemented in pipeline")

    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        from tensorrt_bionemo._trt.layers.transformers import (
            PairformerModule, TokenTransformer)
        return {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        }

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.Boltz2Affinity]


class OpenFold3Factory(ModelComponentsFactory):

    @classmethod
    def get_model_class(cls) -> Type[nn.Module]:
        raise NotImplementedError("OpenFold3 model class not implemented")

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        raise NotImplementedError(
            "OpenFold3 tokenizer not implemented in pipeline")

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        raise NotImplementedError(
            "OpenFold3 feature factory not implemented in pipeline")

    @classmethod
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        raise NotImplementedError(
            "OpenFold3 postprocessor not implemented in pipeline")

    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        from tensorrt_bionemo._trt.layers.transformers import (
            OpenFold3DiffusionTransformer, PairformerModule)
        return {
            "pairformer": PairformerModule,
            "token_transformer": OpenFold3DiffusionTransformer,
        }

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.OpenFold3]


def register_all_factories():
    factories = [
        OpenFold2Factory,
        Boltz1Factory,
        Boltz2Factory,
        Boltz2AffinityFactory,
        OpenFold3Factory,
    ]
    for factory in factories:
        ModelRegistry.register_factory(factory)

    logger.info(f"Registered {len(factories)} model factories")


def get_model_class(model_name: str) -> Type[nn.Module]:
    return ModelRegistry.get_model_class(model_name)


def get_tokenizer(model_name: str) -> "TokenizerBase":
    return ModelRegistry.get_tokenizer(model_name)


def get_feature_factory(model_name: str) -> "FeatureFactoryBase":
    return ModelRegistry.get_feature_factory(model_name)


def get_postprocessor(model_name: str) -> Type["PostProcessorBase"]:
    return ModelRegistry.get_postprocessor(model_name)


def get_building_module_class(model_name: str, module_name: str) -> Any:
    return ModelRegistry.get_building_module_class(model_name, module_name)
