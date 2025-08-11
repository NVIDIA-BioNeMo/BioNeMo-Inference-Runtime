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

from tensorrt_bionemo._trt.layers.affinity import AffinityModule
from tensorrt_bionemo._trt.layers.transformers import (EvoformerStack,
                                                       PairformerModule,
                                                       TokenTransformer)

TRT_BUILDING_MODULES_REGISTRY = {}


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
        "boltz-1": {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
        },
        "boltz-2": {
            "structure_pairformer": PairformerModule,
            "confidence_pairformer": PairformerModule,
            "token_transformer": TokenTransformer,
            "affinity_module": AffinityModule,
        },
        "openfold2": {
            "evoformer": EvoformerStack,
        }
    })


def get_building_module_class(model_name: str, module_name: str):
    """Get a registered module class for a specific model and module type"""
    return TRT_BUILDING_MODULES_REGISTRY.get(model_name, {}).get(module_name)
