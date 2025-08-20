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
from tensorrt_llm.logger import logger

from tensorrt_bionemo.runtime import BaseContextMemoryManager

from ..helper import AcceleratedModules, build_optimized_module
from .convert import (convert_hf_evoformer_torch,
                      convert_hf_extra_msa_stack_torch)
from .modules import EvoformerStackBackendBuilder, ExtraMSAStackBackendBuilder


class OpenFold2AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return ["evoformer", "extra_msa_stack"]


class OpenFold2:

    @staticmethod
    def optimize(
            model: nn.Module,
            accelerated_modules: Optional[OpenFold2AcceleratedModules] = None,
            context_memory_allocator: Optional[BaseContextMemoryManager] = None,
            model_name: str = "openfold2_ptm_1") -> nn.Module:
        """
        This function is used to build the optimized version of Boltz2 model from the original.
        Args:
            model: The original model to be optimized.
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The OpenFold2 optimized model.
        """
        if accelerated_modules is None:
            return model

        module_names = accelerated_modules.get_module_names()
        device = next(model.parameters()).device
        state_dict = model.state_dict()
        opt_m = {}

        if "evoformer" in module_names:
            config = accelerated_modules.get_module_config("evoformer")
            evoformer = build_optimized_module(
                state_dict=state_dict,
                module_name="evoformer",
                backend_builder=EvoformerStackBackendBuilder,
                checkpoint_dir=config.checkpoint,
                backend=config.backend,
                compile=config.compile,
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=config.default,
                convert_weights_func=convert_hf_evoformer_torch,
                convert_weights_func_kwargs={
                    "config": config.default,
                    "mapping": None,
                    "local_checkpoint": config.checkpoint,
                    "model_name": model_name,
                    "weights": state_dict,
                },
            )
            if config.warmup:
                logger.info(
                    "Warming up evoformer module. It can take some minutes...")
                evoformer.warmup()
                logger.info("Warming up evoformer module done.")
            setattr(model, "evoformer", evoformer)
            opt_m["evoformer"] = evoformer

        if "extra_msa_stack" in module_names:
            config = accelerated_modules.get_module_config("extra_msa_stack")
            extra_msa_stack = build_optimized_module(
                state_dict=state_dict,
                module_name="extra_msa_stack",
                backend_builder=ExtraMSAStackBackendBuilder,
                checkpoint_dir=config.checkpoint,
                backend=config.backend,
                compile=config.compile,
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=config.default,
                convert_weights_func=convert_hf_extra_msa_stack_torch,
                convert_weights_func_kwargs={
                    "config": config.default,
                    "mapping": None,
                    "local_checkpoint": config.checkpoint,
                    "model_name": model_name,
                    "weights": state_dict,
                },
            )
            if config.warmup:
                logger.info(
                    "Warming up extra_msa_stack module. It can take some minutes..."
                )
                extra_msa_stack.warmup()
                logger.info("Warming up extra_msa_stack module done.")
            setattr(model, "extra_msa_stack", extra_msa_stack)
            opt_m["extra_msa_stack"] = extra_msa_stack

        return model, opt_m
