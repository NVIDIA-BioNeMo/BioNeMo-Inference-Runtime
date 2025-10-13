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
from tensorrt_bionemo._torch.layers.transformers import PairformerModule
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import (dtype_context, ensure_contiguous,
                                           get_closest_n)

from ..configs import PairformerConfig

# TODO: this is very similar to the boltz1 pairformer, but the mask is called single_mask instead of mask
# in the future, we should merge the two pairformers and remove this class


class PairformerTorch(BackendBase):
    IMPL_CLASS = PairformerModule

    def __init__(self, config: PairformerConfig, impl: nn.Module = None):
        super().__init__(config, impl)

        pairwise_metadata_cls = get_attention_backend(
            config.pairwise_attn_backend).Metadata
        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadatas = {
            "triangle_attn": triangle_metadata_cls(mapping=config.mapping),
            "pairwise_attn": pairwise_metadata_cls(mapping=config.mapping),
        }

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                single_mask: torch.Tensor, pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=s.dtype) as cast_func:
            if self.config.triangle_attn_backend == "TRIFAST":
                self.attn_metadatas["triangle_attn"].closest_n = get_closest_n(
                    s.shape[1] // self.config.mapping.dcp_size)
            s, z = cast_func(self._module)(s,
                                           z,
                                           single_mask,
                                           pair_mask,
                                           attn_metadatas=self.attn_metadatas)
        return s, z


class PairformerTRT(BackendBase):
    IMPL_CLASS = None

    def __init__(self,
                 config: PairformerConfig,
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         impl,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)

    @ensure_contiguous
    def forward(self, s: torch.Tensor, z: torch.Tensor,
                single_mask: torch.Tensor, pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the inputs are contiguous
        if not self.config.support_batch:
            s = s.squeeze(0)
            z = z.squeeze(0)
            single_mask = single_mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)
        original_dtype = s.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "s": s.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "mask": single_mask.to(self.config.torch_dtype),
            "pair_mask": pair_mask.to(self.config.torch_dtype)
        }

        # Use the allocator from the base class for execution
        allocator = self._context_memory_allocator

        outputs = allocator.forward(self, inputs)
        s = outputs["output_s"].to(original_dtype)
        z = outputs["output_z"].to(original_dtype)
        if not self.config.support_batch:
            s = s.unsqueeze(0)
            z = z.unsqueeze(0)
        return s, z


class PairformerBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TRT: PairformerTRT,
        BackendType.TORCH: PairformerTorch,
    }
    CONFIG_CLASS = PairformerConfig
