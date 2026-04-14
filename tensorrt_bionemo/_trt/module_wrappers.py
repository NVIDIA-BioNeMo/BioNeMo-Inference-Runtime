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
    def forward_udf(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
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

    def forward_udf(self,
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
                          z: torch.Tensor = None,
                          mask: torch.Tensor = None,
                          **kwargs) -> torch.Tensor:
        # Incoming shapes from DiffusionModule (B=1 in practice):
        #
        #          Boltz (v2, bias_proj=False)       OF3 (v1, bias_proj=True)
        # a:       [B, mult, N, 768]       4D       [B, mult, N, 768]       4D
        # s:       [B, mult, N, 768]       4D       [B, N, 384]             3D  (no mult dim)
        # z:       [B, 1, N, N, L*H=384]   5D       [B, N, N, 128]          4D  (no prepended 1)
        # mask:    [B, 1, N]               3D       [B, 1, N]               3D
        #
        # TRT engine expects (per iteration):
        #   a: (mult, N, dim),  s: (mult, N, dim_single_cond),  mask: (mult, N)
        #   v2 (Boltz): z: (1, N, N, num_heads * num_blocks)  — pre-projected bias
        #   v1 (OF3):   z: (1, N, N, dim_pairwise)            — raw pair repr
        diffusion_samples = a.shape[1]
        max_diffusion_samples = self.config.multiplicity

        a = a.squeeze(0)  # [DS, N, D]

        # Boltz s is 4D [B, mult, N, D] (per-sample); OF3 s is 3D [B, N, D].
        # For Boltz with mult>1 the squeeze(1) is a no-op leaving s 4D,
        # so we detect the per-sample case by ndim and slice instead of repeat.
        s_per_sample = (s.ndim == 4 and s.shape[1] > 1)
        if s_per_sample:
            s = s.squeeze(0)  # [DS, N, D]
        else:
            # OF3 3D [B,N,D]: squeeze(1) is no-op → stays [1,N,D]
            # Boltz 4D [B,1,N,D]: squeeze(1) removes singleton → [1,N,D]
            # Keep dim-0 so repeat_interleave(n,0) → [n,N,D]
            s = s.squeeze(1)

        # Boltz z is 5D [B,1,N,N,D] → squeeze to 4D; OF3 z is already 4D.
        if z.ndim == 5:
            z = z.squeeze(1)

        # mask is [B,1,N] for both models → squeeze dim-1 → [B,N]
        mask = mask.squeeze(1)

        original_dtype = s.dtype
        outputs = []
        niters = (diffusion_samples + max_diffusion_samples -
                  1) // max_diffusion_samples
        for i in range(niters):
            lo = i * max_diffusion_samples
            hi = min(lo + max_diffusion_samples, diffusion_samples)
            n_repeat = hi - lo

            a_i = a[lo:hi]

            if s_per_sample:
                s_i = s[lo:hi]
            else:
                s_i = s.repeat_interleave(n_repeat, 0)

            mask_i = mask.repeat_interleave(n_repeat, 0)

            inputs = {
                "a": a_i.to(self.config.torch_dtype),
                "s": s_i.to(self.config.torch_dtype),
                "z": z.to(self.config.torch_dtype),
                "mask": mask_i.to(self.config.torch_dtype),
            }

            allocator = self._context_memory_allocator
            outputs.append(allocator.forward(self, inputs)["output_a"])
        outputs = torch.cat(outputs, dim=0)
        return outputs.to(original_dtype).unsqueeze(0)


class EvoformerStackTRT(BackendBase):
    CONFIG_CLASS = EvoformerStackConfig

    def __init__(self,
                 config: EvoformerStackConfig,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.dtype = config.torch_dtype

    @ensure_contiguous
    def forward_udf(
            self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
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
