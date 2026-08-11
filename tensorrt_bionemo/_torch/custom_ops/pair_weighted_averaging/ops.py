# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pair-weighted-averaging fallback and backend selection."""

from __future__ import annotations

import sys
from collections.abc import Callable

import torch

from tensorrt_bionemo.utils import get_sm_version

from ._config import _KERNEL_CM, _KERNEL_D, _KERNEL_H, _SUPPORTED_SM, is_supported_dims
from .cutedsl import PairWeightedAveragingCuTe

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _invoke_vanilla_pwa(
    w: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    Wo: torch.Tensor,
) -> torch.Tensor:
    """Run the unfused PyTorch PWA reference with fp32 accumulation."""
    output_dtype = w.dtype
    H = w.shape[1]
    batch, sequence, n_value, hidden = v.shape
    D = hidden // H
    values = v.reshape(batch, sequence, n_value, H, D)
    j_extent = min(w.shape[-1], n_value)
    output = torch.einsum(
        "bhij,bsjhd->bhsid",
        w[..., :j_extent].float(),
        values[:, :, :j_extent].float(),
    )
    batch, heads, sequence, i_extent, head_dim = output.shape
    output = output.permute(0, 2, 3, 1, 4).reshape(batch, sequence, i_extent, heads * head_dim)
    output = torch.sigmoid(g.float()) * output
    return (output @ Wo.float().t()).to(output_dtype)


_pwa_cute_instance: PairWeightedAveragingCuTe | None = None


def _facade_override(name: str, default: object) -> object:
    """Read compatibility attributes monkeypatched on the package facade."""
    facade = sys.modules.get(__package__)
    return getattr(facade, name, default) if facade is not None else default


def get_pair_weighted_averaging_op(
    dtype: torch.dtype,
    D: int = _KERNEL_D,
    c_m: int = _KERNEL_CM,
    H: int = _KERNEL_H,
) -> Callable:
    """Return the CuTe backend for its supported dtype, SM, and (H, D, c_m) tuple."""
    sm_provider = _facade_override("get_sm_version", get_sm_version)
    sm = sm_provider()
    if sm in _SUPPORTED_SM and dtype in _SUPPORTED_DTYPES and is_supported_dims(H, D, c_m):
        global _pwa_cute_instance
        facade_instance = _facade_override("_pwa_cute_instance", _pwa_cute_instance)
        if facade_instance is not _pwa_cute_instance:
            _pwa_cute_instance = facade_instance
        if _pwa_cute_instance is None:
            _pwa_cute_instance = PairWeightedAveragingCuTe()
        facade = sys.modules.get(__package__)
        if facade is not None:
            facade._pwa_cute_instance = _pwa_cute_instance
        return _pwa_cute_instance
    return _invoke_vanilla_pwa
