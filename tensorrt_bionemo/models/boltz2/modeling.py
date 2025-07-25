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
from typing import Optional

import torch.nn as nn

from tensorrt_bionemo.models.common import AcceleratedModules
from tensorrt_bionemo.modules import (AffinityBackendBuilder,
                                      PairformerBackendBuilder,
                                      TokenTransformerBackendBuilder)
from tensorrt_bionemo.runtime import BaseContextMemoryManager


class Boltz2AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer", "token_transformer"
        ]


class Boltz2AffinityAcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer",
            "token_transformer", "affinity_module1", "affinity_module2"
        ]


class Boltz2:

    @staticmethod
    def optimize(
        model: nn.Module,
        accelerated_modules: Boltz2AcceleratedModules,
        context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ) -> nn.Module:
        """
        This function is used to build the optimized version of Boltz2 model from the original.
        Args:
            model: The original model to be optimized.
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The Boltz2 optimized model.
        """
        if accelerated_modules is None:
            return model
        module_names = accelerated_modules.get_module_names()
        if "structure_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "structure_pairformer")
            backend = accelerated_modules.get_module_backend(
                "structure_pairformer")
            structure_pairformer = PairformerBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model, "pairformer_module", structure_pairformer)

        if "confidence_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "confidence_pairformer")
            backend = accelerated_modules.get_module_backend(
                "confidence_pairformer")
            confidence_pairformer = PairformerBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model.confidence_module, "pairformer_module",
                    confidence_pairformer)

        if "token_transformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "token_transformer")
            backend = accelerated_modules.get_module_backend(
                "token_transformer")
            token_transformer = TokenTransformerBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model.structure_module.score_model, "token_transformer",
                    token_transformer)

        return model


class Boltz2Affinity:

    @staticmethod
    def optimize(
        model: nn.Module,
        accelerated_modules: Boltz2AffinityAcceleratedModules,
        context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ) -> nn.Module:
        """
        This function is used to build the optimized version of Boltz2-Affinity model from the original.
        Args:
            model: The original model to be optimized.
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The Boltz2-Affinity optimized model.
        """
        if accelerated_modules is None:
            return model

        module_names = accelerated_modules.get_module_names()
        if "structure_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "structure_pairformer")
            backend = accelerated_modules.get_module_backend(
                "structure_pairformer")
            structure_pairformer = PairformerBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model, "pairformer_module", structure_pairformer)

        if "confidence_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "confidence_pairformer")
            backend = accelerated_modules.get_module_backend(
                "confidence_pairformer")
            confidence_pairformer = PairformerBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model.confidence_module, "pairformer_module",
                    confidence_pairformer)

        if "token_transformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "token_transformer")
            backend = accelerated_modules.get_module_backend(
                "token_transformer")
            token_transformer = TokenTransformerBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model.structure_module.score_model, "token_transformer",
                    token_transformer)

        if "affinity_module1" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "affinity_module1")
            backend = accelerated_modules.get_module_backend("affinity_module1")
            affinity_module1 = AffinityBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model.affinity_module1, "affinity_module1",
                    affinity_module1)

        if "affinity_module2" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "affinity_module2")
            backend = accelerated_modules.get_module_backend("affinity_module2")
            affinity_module2 = AffinityBackendBuilder.build(
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                context_memory_allocator=context_memory_allocator)
            setattr(model.affinity_module2, "affinity_module2",
                    affinity_module2)

        return model
