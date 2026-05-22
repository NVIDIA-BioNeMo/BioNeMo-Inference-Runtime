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

import pytest
import torch

_CUTEDSL_SUPPORTED_SM = (80, 86, 89, 90)

SM_VERSION: int = (torch.cuda.get_device_capability()[0] * 10 +
                   torch.cuda.get_device_capability()[1]
                   if torch.cuda.is_available() else 0)

SKIP_CUTEDSL_REASON = (
    f"CuTeDSL kernel requires SM{'/'.join(str(s) for s in _CUTEDSL_SUPPORTED_SM)} "
    f"(current SM{SM_VERSION})")

skip_cutedsl = pytest.mark.skipif(
    SM_VERSION not in _CUTEDSL_SUPPORTED_SM,
    reason=SKIP_CUTEDSL_REASON,
)


def skip_if_no_cutedsl():
    """Call inside a test body to skip when CuTeDSL is unsupported."""
    if SM_VERSION not in _CUTEDSL_SUPPORTED_SM:
        pytest.skip(SKIP_CUTEDSL_REASON)


def skip_if_cutedsl(backend_name: str):
    """Skip if *backend_name* is ``"CuTeDSL"`` and the GPU doesn't support it."""
    if backend_name == "CuTeDSL" and SM_VERSION not in _CUTEDSL_SUPPORTED_SM:
        pytest.skip(SKIP_CUTEDSL_REASON)


def make_left_aligned_mask(
    *shape: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    n_valid: torch.Tensor | None = None,
    min_valid: int = 1,
) -> torch.Tensor:
    """Return a left-aligned 0/1 mask of the given shape (last dim is the
    masked axis).

    For each leading-dim slice, the last dim is filled with ``1`` for the
    first ``n`` positions and ``0`` afterwards, with ``n`` drawn uniformly
    from ``[min_valid, last_dim]`` (or supplied via ``n_valid``).

    The CuTeDSL triangle attention left-mask kernel requires this layout,
    and ``pair_mask = seq_mask[..., None] * seq_mask[..., None, :]`` built
    from a left-aligned ``seq_mask`` is also left-aligned.

    Args:
        shape: Tensor shape; the last dim is the axis along which the mask
            is left-aligned.
        dtype: Output dtype (typically ``float32`` or a half dtype).
        device: Output device.
        n_valid: Optional tensor of shape ``shape[:-1]`` giving the count of
            leading 1s per row. If omitted, drawn uniformly per row.
        min_valid: Lower bound for the random count when ``n_valid`` is None.
    """
    assert len(shape) >= 1, "shape must have at least one dimension"
    last = shape[-1]
    leading = shape[:-1] or (1, )
    if n_valid is None:
        n_valid = torch.randint(low=min_valid,
                                high=last + 1,
                                size=leading,
                                device=device,
                                dtype=torch.int64)
    else:
        n_valid = n_valid.to(device=device, dtype=torch.int64)
        assert tuple(n_valid.shape) == tuple(leading), (
            f"n_valid shape {tuple(n_valid.shape)} != leading {tuple(leading)}"
        )
    arange = torch.arange(last, device=device).expand(*leading, last)
    mask = (arange < n_valid.unsqueeze(-1)).to(dtype=dtype)
    return mask.reshape(shape)


def make_left_aligned_pair_mask(
    bs: int,
    n: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    n_valid: torch.Tensor | None = None,
    min_valid: int = 1,
) -> torch.Tensor:
    """Return a ``[bs, n, n]`` pair mask = outer product of a left-aligned
    1D ``seq_mask`` of shape ``[bs, n]``.

    This matches the CuTeDSL triangle-attention left-mask kernel's contract:
    every row ``pair_mask[b, i, :]`` is left-aligned with the same number
    of leading 1s for all valid ``i``, and zero for padded ``i``.
    """
    seq_mask = make_left_aligned_mask(bs,
                                      n,
                                      dtype=dtype,
                                      device=device,
                                      n_valid=n_valid,
                                      min_valid=min_valid)
    return seq_mask[..., None] * seq_mask[..., None, :]
