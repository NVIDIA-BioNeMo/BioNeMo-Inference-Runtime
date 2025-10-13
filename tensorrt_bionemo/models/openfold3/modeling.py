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
from .modules import PairformerBackendBuilder, TokenTransformerBackendBuilder


class OpenFold3AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "pairformer",
            "token_transformer",
        ]


class OpenFold3:

    @staticmethod
    def optimize(
        model: nn.Module,
        accelerated_modules: OpenFold3AcceleratedModules,
        context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ) -> nn.Module:
        """
        This function is used to build the optimized version of OpenFold3 model from the original.
        Args:
            model: The original model to be optimized.
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The OpenFold3 optimized model.
        """
        if accelerated_modules is None:
            return model

        module_names = accelerated_modules.get_module_names()
        device = next(model.parameters()).device
        state_dict = model.state_dict()

        opt_m = {}

        if "pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "pairformer")
            backend = accelerated_modules.get_module_backend("pairformer")
            default_config = accelerated_modules.get_default_module_config(
                "pairformer")
            pairformer = build_optimized_module(
                state_dict=state_dict,
                module_name="pairformer",
                backend_builder=PairformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=None,
                convert_weights_func_kwargs={},
            )
            opt_m["pairformer"] = pairformer
            setattr(model, "pairformer_stack", pairformer)

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
                convert_weights_func=None,
                convert_weights_func_kwargs={},
            )
            opt_m["token_transformer"] = token_transformer
            setattr(model.sample_diffusion.diffusion_module,
                    "diffusion_transformer", token_transformer)

        return model, opt_m
