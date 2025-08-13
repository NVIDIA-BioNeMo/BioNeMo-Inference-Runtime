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

from tensorrt_bionemo.runtime import BaseContextMemoryManager

from ..helper import AcceleratedModules, build_optimized_module
from .convert import (convert_hf_affinity_module_torch,
                      convert_hf_msa_module_torch, convert_hf_pairformer_torch,
                      convert_hf_token_transformer_torch)
from .modules import (AffinityBackendBuilder, MSAModuleBackendBuilder,
                      PairformerBackendBuilder, TokenTransformerBackendBuilder)


class Boltz2AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer",
            "token_transformer", "msa_module"
        ]


class Boltz2AffinityAcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer",
            "token_transformer", "msa_module", "affinity_module1",
            "affinity_module2"
        ]


class Boltz2:

    @staticmethod
    def optimize(
            model: nn.Module,
            accelerated_modules: Boltz2AcceleratedModules,
            context_memory_allocator: Optional[BaseContextMemoryManager] = None,
            is_affinity: bool = False) -> nn.Module:
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
        device = next(model.parameters()).device
        state_dict = model.state_dict()
        opt_m = {}
        model_name = "boltz-2" if not is_affinity else "boltz-2-affinity"

        if "structure_pairformer" in module_names:

            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "structure_pairformer")
            backend = accelerated_modules.get_module_backend(
                "structure_pairformer")
            default_config = accelerated_modules.get_default_module_config(
                "structure_pairformer")
            structure_pairformer = build_optimized_module(
                state_dict=state_dict,
                module_name="structure_pairformer",
                backend_builder=PairformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_pairformer_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "pairformer_type": "structure",
                    "weights": state_dict,
                    "model_name": model_name
                },
            )
            setattr(model, "pairformer_module", structure_pairformer)
            opt_m["structure_pairformer"] = structure_pairformer

        if "confidence_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "confidence_pairformer")
            backend = accelerated_modules.get_module_backend(
                "confidence_pairformer")
            default_config = accelerated_modules.get_default_module_config(
                "confidence_pairformer")
            confidence_pairformer = build_optimized_module(
                state_dict=state_dict,
                module_name="confidence_pairformer",
                backend_builder=PairformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_pairformer_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "pairformer_type": "confidence",
                    "weights": state_dict,
                    "model_name": model_name
                },
            )
            setattr(model.confidence_module, "pairformer_stack",
                    confidence_pairformer)
            opt_m["confidence_pairformer"] = confidence_pairformer

        if "token_transformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "token_transformer")
            backend = accelerated_modules.get_module_backend(
                "token_transformer")
            default_config = accelerated_modules.get_default_module_config(
                "token_transformer")
            token_transformer = build_optimized_module(
                state_dict=state_dict,
                module_name="token_transformer",
                backend_builder=TokenTransformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_token_transformer_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "weights": state_dict,
                    "model_name": model_name
                },
            )
            setattr(model.structure_module.score_model, "token_transformer",
                    token_transformer)
            opt_m["token_transformer"] = token_transformer

        if "msa_module" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "msa_module")
            backend = accelerated_modules.get_module_backend("msa_module")
            default_config = accelerated_modules.get_default_module_config(
                "msa_module")
            msa_module = build_optimized_module(
                state_dict=state_dict,
                module_name="msa_module",
                backend_builder=MSAModuleBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_msa_module_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "weights": state_dict,
                    "model_name": model_name
                },
            )
            setattr(model, "msa_module", msa_module)
            opt_m["msa_module"] = msa_module
        return model, opt_m


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
        device = next(model.parameters()).device
        state_dict = model.state_dict()
        model_name = "boltz-2-affinity"
        model, opt_m = Boltz2.optimize(model,
                                       accelerated_modules,
                                       context_memory_allocator,
                                       is_affinity=True)

        if "affinity_module1" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "affinity_module1")
            backend = accelerated_modules.get_module_backend("affinity_module1")
            default_config = accelerated_modules.get_default_module_config(
                "affinity_module1")
            affinity_module1 = build_optimized_module(
                state_dict=state_dict,
                module_name="affinity_module1",
                backend_builder=AffinityBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_affinity_module_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "weights": state_dict,
                    "affinity_module_name": "affinity_module1",
                    "model_name": model_name
                },
            )
            setattr(model.affinity_module1, "affinity_module1",
                    affinity_module1)
            opt_m["affinity_module1"] = affinity_module1

        if "affinity_module2" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "affinity_module2")
            backend = accelerated_modules.get_module_backend("affinity_module2")
            default_config = accelerated_modules.get_default_module_config(
                "affinity_module2")
            affinity_module2 = build_optimized_module(
                state_dict=state_dict,
                module_name="affinity_module2",
                backend_builder=AffinityBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_affinity_module_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "weights": state_dict,
                    "affinity_module_name": "affinity_module2",
                    "model_name": model_name
                },
            )
            setattr(model.affinity_module2, "affinity_module2",
                    affinity_module2)
            opt_m["affinity_module2"] = affinity_module2

        return model, opt_m
