# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Type

import torch.nn as nn

from tensorrt_bionemo._trt.layers.transformers import (
    EvoformerStack, OpenFold3DiffusionTransformer, PairformerModule,
    TokenTransformer)
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat

TRT_BUILDING_MODULES_REGISTRY = {}
MODEL_REGISTRY = {}


def register_building_module(model_name: str, module_name: str, module_class):
    """Register a module class for a specific model and module type"""
    if model_name not in TRT_BUILDING_MODULES_REGISTRY:
        TRT_BUILDING_MODULES_REGISTRY[model_name] = {}
    TRT_BUILDING_MODULES_REGISTRY[model_name][module_name] = module_class


def register_building_modules(models_modules: dict | list):
    """Register multiple module classes for multiple models at once"""
    if isinstance(models_modules, dict):
        for model_name, modules in models_modules.items():
            if model_name not in TRT_BUILDING_MODULES_REGISTRY:
                TRT_BUILDING_MODULES_REGISTRY[model_name] = {}
            TRT_BUILDING_MODULES_REGISTRY[model_name].update(modules)
    elif isinstance(models_modules, list):
        for model_name, module_name, module_class in models_modules:
            if model_name not in TRT_BUILDING_MODULES_REGISTRY:
                TRT_BUILDING_MODULES_REGISTRY[model_name] = {}
            TRT_BUILDING_MODULES_REGISTRY[model_name][
                module_name] = module_class
    else:
        raise ValueError(
            f"Invalid type for models_modules: {type(models_modules)}")


def register_default_building_modules():
    register_building_modules({
        SupMat.Boltz1: {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        },
        SupMat.Boltz2: {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        },
        SupMat.Boltz2Affinity: {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        },
        SupMat.OpenFold2_FT2: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_FT3: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_FT4: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_FT5: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_NoTempl1: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_NoTempl2: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_NoTempl_PTM1: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_PTM1: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold2_PTM2: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_1: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_2: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_3: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_4: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_5: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_Multimer_1: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_Multimer_2: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_Multimer_3: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_Multimer_4: {
            "evoformer": EvoformerStack,
        },
        SupMat.AlphaFold2_Multimer_5: {
            "evoformer": EvoformerStack,
        },
        SupMat.OpenFold3: {
            "pairformer": PairformerModule,
            "token_transformer": OpenFold3DiffusionTransformer,
        }
    })


def get_building_module_class(model_name: str, module_name: str):
    """Get a registered module class for a specific model and module type"""
    ret = TRT_BUILDING_MODULES_REGISTRY.get(model_name, {}).get(module_name)
    assert ret is not None, f"Module class for {model_name} and {module_name} not found"
    return ret


def register_model(model_name: str, model_class: Type[nn.Module]):
    """Register a model class for a specific model name"""
    if model_name not in MODEL_REGISTRY:
        MODEL_REGISTRY[model_name] = model_class


def register_default_models():
    from tensorrt_bionemo.models.boltz1 import Boltz1
    from tensorrt_bionemo.models.boltz2 import Boltz2, Boltz2Affinity
    from tensorrt_bionemo.models.openfold2 import OpenFold2
    register_model(SupMat.OpenFold2_FT2, OpenFold2)
    register_model(SupMat.OpenFold2_FT3, OpenFold2)
    register_model(SupMat.OpenFold2_FT4, OpenFold2)
    register_model(SupMat.OpenFold2_FT5, OpenFold2)
    register_model(SupMat.OpenFold2_NoTempl1, OpenFold2)
    register_model(SupMat.OpenFold2_NoTempl2, OpenFold2)
    register_model(SupMat.OpenFold2_NoTempl_PTM1, OpenFold2)
    register_model(SupMat.OpenFold2_PTM1, OpenFold2)
    register_model(SupMat.OpenFold2_PTM2, OpenFold2)
    register_model(SupMat.AlphaFold2_1, OpenFold2)
    register_model(SupMat.AlphaFold2_2, OpenFold2)
    register_model(SupMat.AlphaFold2_3, OpenFold2)
    register_model(SupMat.AlphaFold2_4, OpenFold2)
    register_model(SupMat.AlphaFold2_5, OpenFold2)
    register_model(SupMat.AlphaFold2_Multimer_1, OpenFold2)
    register_model(SupMat.AlphaFold2_Multimer_2, OpenFold2)
    register_model(SupMat.AlphaFold2_Multimer_3, OpenFold2)
    register_model(SupMat.AlphaFold2_Multimer_4, OpenFold2)
    register_model(SupMat.AlphaFold2_Multimer_5, OpenFold2)

    register_model(SupMat.Boltz1, Boltz1)
    register_model(SupMat.Boltz2, Boltz2)
    register_model(SupMat.Boltz2Affinity, Boltz2Affinity)


def get_model_class(model_name: str) -> Type[nn.Module]:
    """Get a registered model class for a specific model name"""
    ret = MODEL_REGISTRY.get(model_name)
    assert ret is not None, f"Model class for {model_name} not found"
    return ret
