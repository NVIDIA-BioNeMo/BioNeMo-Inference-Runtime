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

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.transformers import EvoformerStack
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import (dtype_context, ensure_contiguous,
                                           get_closest_n)

from ..configs import EvoformerStackConfig


class EvoformerStackTorch(BackendBase):
    IMPL_CLASS = EvoformerStack

    def __init__(self, config: EvoformerStackConfig, impl: nn.Module = None):
        super().__init__(config, impl)

        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadata = triangle_metadata_cls(mapping=config.mapping)

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m_numdims = m.ndim
        if self.config.support_batch:
            if m_numdims == 3:
                m = m.unsqueeze(0)
                z = z.unsqueeze(0)
                msa_mask = msa_mask.unsqueeze(0)
                pair_mask = pair_mask.unsqueeze(0)
        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=m.dtype) as cast_func:
            if self.config.triangle_attn_backend == "TRIFAST":
                self.attn_metadata.closest_n = get_closest_n(
                    m.shape[1] // self.config.mapping.dcp_size)
            # TODO: support for all_reduce_params
            m, z, s = cast_func(self._module)(m,
                                              z,
                                              msa_mask,
                                              pair_mask,
                                              attn_metadata=self.attn_metadata)

        if self.config.support_batch:
            if m_numdims == 3:
                m = m.squeeze(0)
                z = z.squeeze(0)
                s = s.squeeze(0)
        return m, z, s


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
                **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        BackendType.TORCH: EvoformerStackTorch,
    }
    CONFIG_CLASS = EvoformerStackConfig
