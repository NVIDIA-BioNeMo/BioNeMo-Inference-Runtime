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

import torch

from tensorrt_bionemo.configs import (DiffusionTransformerConfig,
                                      EvoformerStackConfig, PairformerConfig)
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import BackendBase
from tensorrt_bionemo.runtime.misc import ensure_contiguous


class PairformerTRT(BackendBase):
    CONFIG_CLASS = PairformerConfig

    def __init__(self,
                 config: PairformerConfig,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.dtype = config.torch_dtype

    @ensure_contiguous
    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the inputs are contiguous
        if not self.config.support_batch:
            s = s.squeeze(0)
            z = z.squeeze(0)
            mask = mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)
        original_dtype = s.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "s": s.to(self.dtype),
            "z": z.to(self.dtype),
            "mask": mask.to(self.dtype),
            "pair_mask": pair_mask.to(self.dtype)
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


class TokenTransformerTRT(BackendBase):
    CONFIG_CLASS = DiffusionTransformerConfig

    def __init__(self,
                 config: DiffusionTransformerConfig,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.dtype = config.torch_dtype

    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                z: torch.Tensor = None,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        return self._forward_internal(a, s, z, mask, **kwargs)

    @ensure_contiguous
    def _forward_internal(self,
                          a: torch.Tensor,
                          s: torch.Tensor,
                          z: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None,
                          **kwargs) -> torch.Tensor:
        B = a.shape[0]
        assert B == 1, "Batch size must be 1 for token transformer TRT"
        original_dtype = s.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "a": a.to(self.dtype).squeeze(0),
            "s": s.to(self.dtype).squeeze(0),
            "z": z.to(self.dtype).squeeze(0),
            "mask": mask.to(self.dtype).squeeze(0)
        }

        # Use the allocator from the base class for execution
        outputs = self._context_memory_allocator.forward(self, inputs)
        return outputs["output_a"].to(original_dtype).unsqueeze(0)


class EvoformerStackTRT(BackendBase):
    CONFIG_CLASS = EvoformerStackConfig

    def __init__(self,
                 config: EvoformerStackConfig,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.dtype = config.torch_dtype

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
            "m": m.to(self.dtype),
            "z": z.to(self.dtype),
            "msa_mask": msa_mask.to(self.dtype),
            "pair_mask": pair_mask.to(self.dtype)
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
