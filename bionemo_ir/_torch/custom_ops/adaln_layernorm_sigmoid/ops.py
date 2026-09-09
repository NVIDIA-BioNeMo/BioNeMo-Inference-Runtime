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
"""PyTorch fallbacks and dispatch for the AdaLN normalization-sigmoid op."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

from bionemo_ir.utils import get_sm_version

from ._config import _TORCH_TO_CUTLASS_DTYPE, SHIPPED_N, SUPPORTED_SMS

_adaln_lns_cute_instances: dict[bool, Callable] = {}


def _invoke_vanilla_adaln_layernorm_sigmoid(
    x: torch.Tensor,
    s_scale: torch.Tensor,
    s_bias: torch.Tensor,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
    rms_norm: bool = False,
) -> torch.Tensor:
    """Compute the fused operation in PyTorch."""
    normed = (
        torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)
        if rms_norm
        else torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=eps)
    )
    result = (torch.sigmoid(s_scale) * normed + s_bias).to(x.dtype)
    destination = x if out is None else out
    destination.copy_(result.expand_as(destination) if result.shape != destination.shape else result)
    return destination


def get_adaln_layernorm_sigmoid_op(dtype: torch.dtype, N: int, rms_norm: bool = False) -> Callable:
    """Return the fused CuTeDSL LayerNorm or RMSNorm op."""
    if get_sm_version() not in SUPPORTED_SMS or dtype not in _TORCH_TO_CUTLASS_DTYPE or N not in SHIPPED_N:
        if rms_norm:
            return partial(_invoke_vanilla_adaln_layernorm_sigmoid, rms_norm=True)
        return _invoke_vanilla_adaln_layernorm_sigmoid

    from .cutedsl import AdaLNLayerNormSigmoidCuTe

    instance = _adaln_lns_cute_instances.get(rms_norm)
    if instance is None:
        instance = AdaLNLayerNormSigmoidCuTe(rms_norm=rms_norm)
        _adaln_lns_cute_instances[rms_norm] = instance
    return instance
