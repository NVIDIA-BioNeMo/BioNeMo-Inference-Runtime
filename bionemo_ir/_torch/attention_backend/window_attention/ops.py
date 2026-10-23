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
"""Sequence-local (windowed) attention over prepared biases: kernel selection and the call."""

from __future__ import annotations

import functools

import torch

from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryError

from ._config import config_kernel_abi
from .cutedsl import WindowAttentionCuTe

_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_ops: dict[tuple, WindowAttentionCuTe] = {}


@functools.cache
def _sm_version(device_index: int) -> int:
    """Compute capability of one device as an SM number."""
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


def get_window_attention_op(
    dtype: torch.dtype, head_dim: int, n_queries: int, n_keys: int, device_index: int | None = None
) -> WindowAttentionCuTe | None:
    """The kernel for this dtype and window shape on ``device_index`` (default: the current device), or ``None``.

    The kernel is compiled from source, or loaded from the packaged CUBIN, here rather than in the forward pass or
    a CUDA graph capture. ``None`` without CUDA, for a dtype other than bf16 and fp16, or for a GPU and window
    shape without a config or kernel; under ``CUTEDSL_FORCE_CUBIN`` a missing CUBIN raises instead.
    """
    if not torch.cuda.is_available():
        return None
    dtype_str = _DTYPE_STR.get(dtype)
    if dtype_str is None:
        return None
    device = torch.cuda.current_device() if device_index is None else device_index
    sm_version = _sm_version(device)
    key = (sm_version, dtype_str, head_dim, n_queries, n_keys)
    op = _ops.get(key)
    if op is None:
        if config_kernel_abi(sm_version, head_dim, n_queries, n_keys) is None:
            return None
        op = WindowAttentionCuTe(sm_version, dtype_str, head_dim, n_queries, n_keys)
    try:
        with torch.cuda.device(device):
            op.executable()
    except CuTeDSLKernelLibraryError:
        return None
    _ops[key] = op
    return op


def window_attention_supported(
    dtype: torch.dtype, head_dim: int, n_queries: int, n_keys: int, device_index: int | None = None
) -> bool:
    """Whether :func:`window_attention` serves this configuration on ``device_index`` (default: the current device)."""
    return get_window_attention_op(dtype, head_dim, n_queries, n_keys, device_index) is not None


def _aligned(x: torch.Tensor, dims: int) -> bool:
    """16-byte aligned base, positive 16-byte multiple strides on the leading ``dims`` axes, unit last stride."""
    return x.stride(-1) == 1 and x.data_ptr() % 16 == 0 and all(s > 0 and s % 8 == 0 for s in x.stride()[:dims])


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    n_queries: int,
    n_keys: int,
    scale: float | None = None,
) -> torch.Tensor:
    """``softmax(scale * q k^T + bias) v`` over each query window's key window.

    Window ``t`` holds query rows ``[t * n_queries, (t + 1) * n_queries)`` and attends to key rows
    ``[t * n_queries - p, t * n_queries - p + n_keys)``, ``p = (n_keys - n_queries) // 2``; key rows outside
    ``[0, N)`` are zero, and the bias masks them.

    Args:
        q, k, v: ``[S, N, H, D]`` CUDA views with ``D`` contiguous and heads adjacent, ``N`` whole query windows;
            the ``S`` sequences are ``B`` batch items of ``S / B`` consecutive samples.
        bias: ``[B, N / n_queries, H, n_queries, n_keys]`` additive bias that carries the key mask; each batch
            item's samples share it.
        n_queries, n_keys: Query and key window sizes.
        scale: Logit scale, default ``D ** -0.5``.

    Returns:
        ``[S, N, H, D]`` contiguous output in ``q``'s dtype.
    """
    S, N, H, D = q.shape
    op = get_window_attention_op(q.dtype, D, n_queries, n_keys, q.device.index)
    if op is None:
        raise RuntimeError(
            f"no windowed attention kernel for {q.dtype}, D={D}, windows {n_queries}x{n_keys} on {q.device}"
        )
    bias = bias.to(device=q.device, dtype=q.dtype)
    if q.dtype == torch.float16:
        # fp16 turns a -1e9 mask into -inf, and a window whose keys are all masked would give NaN.
        bias = bias.clamp(min=torch.finfo(torch.float16).min)
    # The kernel moves 16-byte rows with heads adjacent; other layouts take a contiguous copy.
    q, k, v = (x if x.stride(2) == D and _aligned(x, 2) else x.contiguous() for x in (q, k, v))
    if not _aligned(bias, 4):
        bias = bias.contiguous()
    out = torch.empty((S, N, H, D), device=q.device, dtype=q.dtype)
    with torch.cuda.device(q.device):
        return op(q, k, v, bias, out, D**-0.5 if scale is None else float(scale))
