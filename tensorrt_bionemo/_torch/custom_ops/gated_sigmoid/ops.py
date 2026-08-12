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
"""PyTorch fallback for the gated sigmoid op."""

from __future__ import annotations

from collections.abc import Callable

import torch

from tensorrt_bionemo.utils import get_sm_version

_SUPPORTED_SM = {80, 86, 89, 90}
_SUPPORTED_NK = {
    (64, 64),
    (128, 128),
    (256, 64),
    (256, 128),
    (256, 256),
    (384, 384),
    (768, 384),
    (768, 768),
}
_gated_sigmoid_cute_instance = None


def _invoke_vanilla_gated_sigmoid(
    s: torch.Tensor,
    weight: torch.Tensor,
    mha_out: torch.Tensor,
    bias: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``sigmoid(s @ W.T [+ bias]) * mha_out`` in PyTorch."""
    N_out = weight.shape[0]
    gate = torch.nn.functional.linear(s, weight, bias).sigmoid()
    if gate.numel() == mha_out.numel():
        result = (gate.reshape(-1, N_out) * mha_out.reshape(-1, N_out)).view_as(mha_out)
    else:
        result = gate * mha_out
    if output is not None:
        output.copy_(result.view_as(output))
        return output
    return result


def get_gated_sigmoid_op(dtype: torch.dtype, N: int = 128, K: int = 128) -> Callable:
    """Return the fused CuTeDSL op when this GPU, dtype, and shape are supported."""
    if (
        get_sm_version() not in _SUPPORTED_SM
        or dtype not in (torch.float16, torch.bfloat16)
        or (N, K) not in _SUPPORTED_NK
    ):
        return _invoke_vanilla_gated_sigmoid

    from .cutedsl import GatedSigmoidCuTe

    global _gated_sigmoid_cute_instance
    if _gated_sigmoid_cute_instance is None:
        _gated_sigmoid_cute_instance = GatedSigmoidCuTe()
    return _gated_sigmoid_cute_instance
