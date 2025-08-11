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

import torch
import torch.nn as nn
from tensorrt_llm._utils import str_dtype_to_trt

from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import ensure_contiguous

from ..configs import EvoformerStackConfig


class EvoformerStackTRT(BackendBase):
    IMPL_CLASS = None

    def __init__(self,
                 config: EvoformerStackConfig,
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         impl,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)

    @ensure_contiguous
    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the inputs are contiguous
        m_numdims = m.ndim
        if self.config.support_batch:
            if m_numdims == 3:
                m = m.unsqueeze(0)
                z = z.unsqueeze(0)
                msa_mask = msa_mask.unsqueeze(0)
                pair_mask = pair_mask.unsqueeze(0)
        original_dtype = m.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "m": m.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "msa_mask": msa_mask.to(self.config.torch_dtype),
            "pair_mask": pair_mask.to(self.config.torch_dtype)
        }

        # Use the allocator from the base class for execution
        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        m = outputs["output_m"].to(original_dtype)
        z = outputs["output_z"].to(original_dtype)
        s = outputs["output_s"].to(original_dtype)
        if self.config.support_batch:
            if m_numdims == 3:
                m = m.squeeze(0)
                z = z.squeeze(0)
                s = s.squeeze(0)
        return m, z, s


class EvoformerStackBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TRT: EvoformerStackTRT,
    }
    CONFIG_CLASS = EvoformerStackConfig
