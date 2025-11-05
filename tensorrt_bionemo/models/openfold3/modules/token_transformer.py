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
from tensorrt_llm import str_dtype_to_trt

from tensorrt_bionemo._torch.attention_backend.utils import (
    AttentionType, get_attention_backend)
from tensorrt_bionemo._torch.layers.transformers import \
    OpenFold3DiffusionTransformer
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import ensure_contiguous

from ..configs import DiffusionTransformerConfig


class TokenTransformerTorch(BackendBase):
    IMPL_CLASS = OpenFold3DiffusionTransformer

    def __init__(self,
                 config: DiffusionTransformerConfig,
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         impl,
                         context_memory_allocator=context_memory_allocator)

    @ensure_contiguous
    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                z: torch.Tensor = None,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        z = z.squeeze(1)
        attn_pairwise_metadata_cls = get_attention_backend(
            self.config.pairwise_attn_backend, AttentionType.PAIRWISE).Metadata
        original_dtype = s.dtype
        output = self._module(
            a,
            s,
            z,
            mask,
            attn_metadata=attn_pairwise_metadata_cls(bias_cache={}))
        return output.to(original_dtype)


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

    @ensure_contiguous
    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                z: torch.Tensor = None,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        diffusion_samples = a.shape[1]
        max_diffusion_samples = self.config.max_diffusion_samples
        # assert diffusion_samples == 1, f"diffusion_samples must be 1, but got {diffusion_samples}"
        # remove the diffusion_samples dimension
        s = s.squeeze(1)
        z = z.squeeze(1)
        mask = mask.squeeze(1)
        a = a.squeeze(0)
        outputs = []
        niters = (diffusion_samples + max_diffusion_samples -
                  1) // max_diffusion_samples
        for i in range(niters):
            a_i = a[i * max_diffusion_samples:(i + 1) * max_diffusion_samples,
                    ...]

            original_dtype = s.dtype

            # TODO: Use config.get_input_names() to get the input names
            inputs = {
                "a": a_i.to(self.config.torch_dtype),
                "s": s.to(self.config.torch_dtype),
                "z": z.to(self.config.torch_dtype),
                "mask": mask.to(self.config.torch_dtype)
            }

            # Use the allocator from the base class for execution
            allocator = self._context_memory_allocator
            outputs.append(allocator.forward(self, inputs)["output_a"])
        outputs = torch.cat(outputs, dim=0)
        return outputs.to(original_dtype).unsqueeze(0)


class TokenTransformerBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TORCH: TokenTransformerTorch,
        BackendType.TRT: TokenTransformerTRT
    }
    CONFIG_CLASS = DiffusionTransformerConfig
