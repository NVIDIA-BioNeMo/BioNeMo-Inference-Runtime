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

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.modules.boltz.trunk import MSAModule
from tensorrt_bionemo.configs import MSAModuleConfig
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import dtype_context


class MSAModuleTorch(BackendBase):
    IMPL_CLASS = MSAModule

    def __init__(self, config: MSAModuleConfig, impl: nn.Module = None):
        super().__init__(config, impl)

        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadata = triangle_metadata_cls(mapping=config.mapping)

    def forward(self, z: torch.Tensor, emb: torch.Tensor,
                feats: dict[str, torch.Tensor],
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=z.dtype,
                           skip_keys=["msa"]) as cast_func:
            z = cast_func(self._module)(z,
                                        emb,
                                        msa=feats["msa"],
                                        has_deletion=feats["has_deletion"],
                                        deletion_value=feats["deletion_value"],
                                        msa_paired=feats["msa_paired"],
                                        msa_mask=feats["msa_mask"],
                                        token_pad_mask=feats["token_pad_mask"],
                                        attn_metadata=self.attn_metadata)
        return z


class MSAModuleBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TORCH: MSAModuleTorch,
    }
    CONFIG_CLASS = MSAModuleConfig
