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
from .convert import (convert_hf_msa_module_torch, convert_hf_pairformer_torch,
                      convert_hf_token_transformer_torch)
from .modules import (MSAModuleBackendBuilder, PairformerBackendBuilder,
                      TokenTransformerBackendBuilder)


class Boltz1AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer",
            "token_transformer", "msa_module"
        ]


class Boltz1:

    @staticmethod
    def optimize(
        model: nn.Module,
        accelerated_modules: Boltz1AcceleratedModules,
        context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ) -> nn.Module:
        """
        This function is used to build the optimized version of Boltz1 model from the original.
        Args:
            model: The original model to be optimized.
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The Boltz1 optimized model.
        """
        if accelerated_modules is None:
            return model

        module_names = accelerated_modules.get_module_names()
        device = next(model.parameters()).device
        state_dict = model.state_dict()

        opt_m = {}

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
                    "weights": state_dict
                },
            )
            opt_m["structure_pairformer"] = structure_pairformer
            setattr(model, "pairformer_module", structure_pairformer)

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
                    "weights": state_dict
                },
            )
            opt_m["confidence_pairformer"] = confidence_pairformer
            setattr(model.confidence_module, "pairformer_module",
                    confidence_pairformer)

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
                    "weights": state_dict
                },
            )
            opt_m["token_transformer"] = token_transformer
            setattr(model.structure_module.score_model, "token_transformer",
                    token_transformer)

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
                    "weights": state_dict
                },
            )
            opt_m["msa_module"] = msa_module
            setattr(model, "msa_module", msa_module)
        return model, opt_m
