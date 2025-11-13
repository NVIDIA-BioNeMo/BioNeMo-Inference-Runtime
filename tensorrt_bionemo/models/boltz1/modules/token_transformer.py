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
from tensorrt_bionemo._torch.layers.transformers import \
    BoltzDiffusionTransformer
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import dtype_context, ensure_contiguous

from ..configs import DiffusionTransformerConfig


class TokenTransformerTorch(BackendBase):
    IMPL_CLASS = BoltzDiffusionTransformer

    def __init__(self,
                 config: DiffusionTransformerConfig,
                 impl: nn.Module = None):
        super().__init__(config, impl)
        self.metadata_cls = get_attention_backend(
            self.config.pairwise_attn_backend).Metadata
        self.attn_metadata = self.metadata_cls(mapping=self.config.mapping)

    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                bias: torch.Tensor = None,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=s.dtype) as cast_func:
            # TODO: add allreduce parameters here
            a = cast_func(self._module)(a,
                                        s,
                                        z=bias,
                                        mask=mask,
                                        attn_metadata=self.attn_metadata)
        return a


class TokenTransformerTRT(BackendBase):
    IMPL_CLASS = None

    def __init__(self,
                 config: DiffusionTransformerConfig,
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         impl,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)

    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                bias: torch.Tensor = None,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        return self._forward_internal(a, s, bias, mask, **kwargs)

    @ensure_contiguous
    def _forward_internal(self,
                          a: torch.Tensor,
                          s: torch.Tensor,
                          z: torch.Tensor = None,
                          mask: torch.Tensor = None,
                          **kwargs) -> torch.Tensor:
        original_dtype = s.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "a": a.to(self.config.torch_dtype),
            "s": s.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "mask": mask.to(self.config.torch_dtype)
        }

        # Use the allocator from the base class for execution
        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        return outputs["output_a"].to(original_dtype)


class TokenTransformerBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TORCH: TokenTransformerTorch,
        BackendType.TRT: TokenTransformerTRT,
    }
    CONFIG_CLASS = DiffusionTransformerConfig
