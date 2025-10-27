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

from typing import Callable, Optional

import torch
import torch.nn as nn
from tensorrt_llm._utils import str_dtype_to_trt

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.aux_heads.boltz import (
    AffinityModule, compute_distogram, create_cross_pair_mask)
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import ensure_contiguous

from ..configs import AffinityModuleConfig


class AffinityModuleTorch(BackendBase):
    # Boltz2 affinity module
    IMPL_CLASS = AffinityModule

    def __init__(self,
                 config: AffinityModuleConfig,
                 load_weights_fn: Optional[Callable] = None,
                 impl: nn.Module = None):
        super().__init__(config, load_weights_fn, impl)

        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadatas = {
            "triangle_attn": triangle_metadata_cls(mapping=config.mapping),
        }

        boundaries = torch.linspace(2, config.max_dist,
                                    config.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)

    def forward(self,
                s_inputs: torch.Tensor,
                z: torch.Tensor,
                x_pred: torch.Tensor,
                feats: dict[str, torch.Tensor],
                multiplicity=1,
                **kwargs) -> dict[str, torch.Tensor]:
        # Sanity check
        assert multiplicity == 1, "Multiplicity > 1 is not supported"
        assert "token_to_rep_atom" in feats, "token_to_rep_atom is required"
        assert "pad_token_mask" in feats, "pad_token_mask is required"
        assert "mol_type" in feats, "mol_type is required"
        assert "affinity_token_mask" in feats, "affinity_token_mask is required"

        distogram = compute_distogram(x_pred, self.boundaries,
                                      feats["token_to_rep_atom"], multiplicity)
        cross_pair_mask_0, cross_pair_mask_1 = \
                    create_cross_pair_mask(
                                    feats["pad_token_mask"],
                                    feats["mol_type"],
                                    feats["affinity_token_mask"],
                                    multiplicity,
                                    include_mask_for_head=True)

        original_dtype = s_inputs.dtype
        # Cast the output to the expected dtype
        pred_value, logits_binary = self._module(
            s_inputs.to(self.config.torch_dtype), z.to(self.config.torch_dtype),
            distogram.to(torch.int32),
            cross_pair_mask_0.to(self.config.torch_dtype),
            cross_pair_mask_1.to(self.config.torch_dtype))
        return {
            "affinity_pred_value": pred_value.to(original_dtype),
            "affinity_logits_binary": logits_binary.to(original_dtype)
        }


class AffinityModuleTRT(BackendBase):
    IMPL_CLASS = None

    def __init__(self,
                 config: AffinityModuleConfig,
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         impl,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)
        self.int32_dtype = str_dtype_to_trt("int32")
        boundaries = torch.linspace(2, config.max_dist,
                                    config.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)

    @ensure_contiguous
    def forward(self,
                s_inputs: torch.Tensor,
                z: torch.Tensor,
                x_pred: torch.Tensor,
                feats: dict[str, torch.Tensor],
                multiplicity=1,
                **kwargs) -> dict[str, torch.Tensor]:
        # Ensure the inputs are contiguous
        original_dtype = s_inputs.dtype

        distogram = compute_distogram(x_pred, self.boundaries,
                                      feats["token_to_rep_atom"], multiplicity)
        cross_pair_mask_0, cross_pair_mask_1 = \
                    create_cross_pair_mask(
                                    feats["token_pad_mask"],
                                    feats["mol_type"],
                                    feats["affinity_token_mask"],
                                    multiplicity,
                                    include_mask_for_head=True)

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "s": s_inputs.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "distogram": distogram.to(torch.int32),
            "cross_pair_mask_0": cross_pair_mask_0.to(self.config.torch_dtype),
            "cross_pair_mask_1": cross_pair_mask_1.to(self.config.torch_dtype)
        }

        # Use the allocator from the base class for execution
        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        pred_value = outputs["pred_value"].to(original_dtype)
        logits_binary = outputs["logits_binary"].to(original_dtype)
        return {
            "affinity_pred_value": pred_value,
            "affinity_logits_binary": logits_binary
        }


class AffinityBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TRT: AffinityModuleTRT,
        BackendType.TORCH: AffinityModuleTorch,
    }
    CONFIG_CLASS = AffinityModuleConfig
