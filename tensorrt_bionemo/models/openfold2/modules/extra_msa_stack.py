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
import bisect
import itertools
import math

import torch
import torch.nn as nn
from tensorrt_llm.logger import logger
from tqdm import tqdm

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.modules.openfold.trunk import ExtraMSAStack
from tensorrt_bionemo._torch.utils import pad_dim
from tensorrt_bionemo.configs import ExtraMSAStackConfig
from tensorrt_bionemo.runtime.backend import (BackendBase, BackendBuilder,
                                              BackendType)
from tensorrt_bionemo.runtime.misc import dtype_context


class ExtraMSAStackTorch(BackendBase):
    IMPL_CLASS = ExtraMSAStack

    def __init__(self, config: ExtraMSAStackConfig, impl: nn.Module = None):
        super().__init__(config, impl)

        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadata = triangle_metadata_cls(mapping=config.mapping)

        nres_fl2 = int(math.floor(math.log2(self.config.max_seq_len)))
        nseq_fl2 = int(math.floor(math.log2(self.config.max_msa_size)))

        n_res_list = []
        for i in range(5, nres_fl2):
            n_res_list.append(2**i)
        n_res_list = [self.config.max_seq_len]
        additional = [384, 768, 1280, 1536, 1792]
        for v in additional:
            if v <= self.config.max_seq_len:
                n_res_list.append(v)
        self.n_res_list = sorted(list(set(n_res_list)))

        n_seq_list = []
        for i in range(9, nseq_fl2):
            n_seq_list.append(2**i)
        n_seq_list.extend([1536, self.config.max_msa_size])
        self.n_seq_list = sorted(list(set(n_seq_list)))

    def warmup(self):
        with torch.no_grad():
            for n_res, n_seq in tqdm(itertools.product(self.n_res_list,
                                                       self.n_seq_list),
                                     desc="Warmup"):
                try:
                    m = torch.randn(self.config.max_batch_size,
                                    n_seq,
                                    n_res,
                                    self.config.c_m,
                                    dtype=self.config.torch_dtype,
                                    device="cuda")
                    z = torch.randn(self.config.max_batch_size,
                                    n_res,
                                    n_res,
                                    self.config.c_z,
                                    dtype=self.config.torch_dtype,
                                    device="cuda")
                    msa_mask = torch.randint(
                        0,
                        2, (self.config.max_batch_size, n_seq, n_res),
                        dtype=self.config.torch_dtype,
                        device="cuda")
                    pair_mask = torch.randint(
                        0,
                        2, (self.config.max_batch_size, n_res, n_res),
                        dtype=self.config.torch_dtype,
                        device="cuda")
                    self._module(m,
                                 z,
                                 msa_mask,
                                 pair_mask,
                                 attn_metadata=self.attn_metadata)
                except torch.OutOfMemoryError:
                    logger.warning(
                        f"Out of memory at n_res={n_res}, n_seq={n_seq}")
                    continue

    def _pad_inputs(
        self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ TODO: consider when world_size > 1 """
        if len(self.n_res_list) == 0 or len(self.n_seq_list) == 0:
            return m, z, msa_mask, pair_mask, 0

        assert self.config.max_batch_size == 1, "max_batch_size must be 1"
        n_seq = m.shape[1]
        n_res = m.shape[2]
        ref_n_res = n_res
        ref_n_seq = msa_mask.sum(dim=-2).max().item()
        n_seq_idx = bisect.bisect_left(self.n_seq_list, ref_n_seq)
        n_res_idx = bisect.bisect_left(self.n_res_list, ref_n_res)

        pad_n_seq = self.n_seq_list[n_seq_idx]
        pad_n_res = self.n_res_list[n_res_idx]

        if pad_n_seq > n_seq:
            m = pad_dim(m, 1, pad_n_seq - n_seq)
            msa_mask = pad_dim(msa_mask, 1, pad_n_seq - n_seq)
        else:
            m = m[:, :pad_n_seq, ...]
            msa_mask = msa_mask[:, :pad_n_seq, ...]

        if pad_n_res > n_res:
            m = pad_dim(m, 2, pad_n_res - n_res)
            z = pad_dim(z, 1, pad_n_res - n_res)
            z = pad_dim(z, 2, pad_n_res - n_res)
            msa_mask = pad_dim(msa_mask, 2, pad_n_res - n_res)
            pair_mask = pad_dim(pair_mask, 1, pad_n_res - n_res)
            pair_mask = pad_dim(pair_mask, 2, pad_n_res - n_res)

        return m.contiguous(), z.contiguous(), msa_mask.contiguous(
        ), pair_mask.contiguous(), n_res - pad_n_res

    def _unpad_outputs(self, z: torch.Tensor,
                       offset_n_res: int) -> torch.Tensor:
        if offset_n_res == 0:
            return z
        n_res = z.shape[1]
        z = z[:, :n_res + offset_n_res, :n_res + offset_n_res, ...]
        return z

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        m_numdims = m.ndim
        if self.config.support_batch:
            if m_numdims == 3:
                m = m.unsqueeze(0)
                z = z.unsqueeze(0)
                msa_mask = msa_mask.unsqueeze(0)
                pair_mask = pair_mask.unsqueeze(0)

        offset_n_res = 0
        if self.config.padding_inputs:
            m, z, msa_mask, pair_mask, offset_n_res = self._pad_inputs(
                m, z, msa_mask, pair_mask)

        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=m.dtype) as cast_func:
            # TODO: support for all_reduce_params
            z = cast_func(self._module)(m,
                                        z,
                                        msa_mask,
                                        pair_mask,
                                        attn_metadata=self.attn_metadata)

        if self.config.padding_inputs:
            z = self._unpad_outputs(z, offset_n_res)

        if self.config.support_batch:
            if m_numdims == 3:
                z = z.squeeze(0)
        return z


class ExtraMSAStackBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        BackendType.TORCH: ExtraMSAStackTorch,
    }
    CONFIG_CLASS = ExtraMSAStackConfig
