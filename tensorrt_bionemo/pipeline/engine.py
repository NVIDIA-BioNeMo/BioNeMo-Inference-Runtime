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
from typing import Any, Callable, Optional, Type

import numpy as np
import torch
import torch.nn as nn

from tensorrt_bionemo.configs.base import EngineConfig
from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.pipeline.base import PostProcessorBase
from tensorrt_bionemo.runtime import OnDemandContextMemoryManager


class FoldingEngine:
    """ Folding engine for folding tasks. """

    def __init__(self,
                 config: EngineConfig,
                 model_cls: Type[nn.Module],
                 postprocessor_cls: Optional[Type[Callable]] = None) -> None:
        self.config = config
        self.model_name = config.name
        self.model_config = config.model
        self.device_config = config.device
        self.postprocessor_config = config.postprocessor
        self.accelerated_configs = config.accelerated
        self.model_cls = model_cls
        if postprocessor_cls is None:
            self.postprocessor_cls = PostProcessorBase
        else:
            self.postprocessor_cls = postprocessor_cls

        self.create_model()
        self.create_postprocessor()

    def create_model(self) -> nn.Module:
        """ Create the model from the model config. """
        # Need the model name to load the weights
        self.model = self.model_cls(config=self.model_config,
                                    model_name=self.model_name)
        self.model.to(self.device_config.device)
        self.model.eval()

        self.context_memory_allocator = None
        if self.accelerated_configs is not None:
            self.context_memory_allocator = OnDemandContextMemoryManager()
            self.model, _ = self.model.optimize(
                self.accelerated_configs,
                context_memory_allocator=self.context_memory_allocator)
        return self.model

    def create_postprocessor(self) -> Callable:
        """ Create the post processor from the post processor config. """
        self.postprocessor = self.postprocessor_cls(self.postprocessor_config)

    def transfer_batch_to_device(self, batch: dict[str,
                                                   Any]) -> dict[str, Any]:
        """ Transfer the batch to the device. """
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                device_batch[k] = torch.as_tensor(
                    v, device=self.device_config.device)
            else:
                device_batch[k] = v
        return device_batch

    @torch.inference_mode()
    def execute(self, batch: dict[str, Any]) -> FoldingOutput:
        """ Execute the model with the input. """
        device_batch = self.transfer_batch_to_device(batch)
        output = self.model(device_batch)
        output = self.postprocessor(device_batch, output)
        return output
