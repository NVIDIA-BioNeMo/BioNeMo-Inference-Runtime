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

SM_VERSION: int = (torch.cuda.get_device_capability()[0] * 10 +
                   torch.cuda.get_device_capability()[1]
                   if torch.cuda.is_available() else 0)

# The generic CuTeDSL SM range (Ampere through Hopper). Most kernels run on
# the whole range; ``skip_if_no_cutedsl`` / ``skip_if_cutedsl`` look up an
# op-specific override below by op name when one is supplied.
_CUTEDSL_SUPPORTED_SM = (80, 86, 89, 90)

# Per-op CuTeDSL SM overrides. Empty today: every CuTeDSL op was verified to
# run on the whole generic range from its dispatch / config coverage in
# ``tensorrt_bionemo/_torch``:
#   * gated_sigmoid            -- get_gated_sigmoid_op gates on
#                                 ``sm in (80, 86, 89, 90)``.
#   * dual_gemm_x_x / x0_x1     -- get_dual_gemm_*_op use
#                                 ``_TUNED_SMS = (80, 86, 89, 90)``; JSON
#                                 configs exist for all four (the SM90
#                                 ping-pong is just the kernel the ``sm90``
#                                 configs resolve to, not a distinct op).
#   * adaln_layernorm_sigmoid   -- runs on every SM (``sm < 90`` uses the
#                                 default single-bucket schedule).
#   * triangle_attention /      -- left-mask kernels ship Ampere (SM80) +
#     pairwise_attention           Hopper (SM90) classes with configs for
#                                 80/86/89/90.
# Add an entry only when an op's kernel support genuinely diverges from the
# generic range; ``skip_if_no_cutedsl`` / ``skip_if_cutedsl`` then honor it.
# (Tests that assert an SM-specific *kernel path* should use a direct SM
# gate like ``skip_if_not_sm90`` instead -- that's a test requirement, not an
# op-support fact.)
_CUTEDSL_OP_SUPPORTED_SM: "dict[str, tuple[int, ...]]" = {}


def _cutedsl_supported_sm(op_name: str | None = None) -> tuple[int, ...]:
    """SM versions the CuTeDSL kernel for *op_name* supports.

    Unknown / ``None`` op names fall back to the generic CuTeDSL range.
    """
    return _CUTEDSL_OP_SUPPORTED_SM.get(op_name, _CUTEDSL_SUPPORTED_SM)


def _cutedsl_skip_reason(supported: tuple[int, ...]) -> str:
    return (f"CuTeDSL kernel requires SM{'/'.join(str(s) for s in supported)} "
            f"(current SM{SM_VERSION})")


SKIP_CUTEDSL_REASON = _cutedsl_skip_reason(_CUTEDSL_SUPPORTED_SM)

skip_cutedsl = pytest.mark.skipif(
    SM_VERSION not in _CUTEDSL_SUPPORTED_SM,
    reason=SKIP_CUTEDSL_REASON,
)


def skip_if_no_cutedsl(op_name: str | None = None):
    """Call inside a test body to skip when the current GPU can't run the
    CuTeDSL kernel for *op_name*.

    Args:
        op_name: Optional op identifier. Architecture-specific kernels (e.g.
            ``"dual_gemm_sm90"``, the Hopper-only ping-pong dual GEMM) are
            checked against their own SM set; ``None`` or any unlisted name
            uses the generic CuTeDSL range.
    """
    supported = _cutedsl_supported_sm(op_name)
    if SM_VERSION not in supported:
        pytest.skip(_cutedsl_skip_reason(supported))


def skip_if_cutedsl(backend_name: str, op_name: str | None = None):
    """Skip if *backend_name* is ``"CuTeDSL"`` and the current GPU can't run
    the CuTeDSL kernel for *op_name*.

    Args:
        backend_name: Attention / module backend selected by the test; the
            check is a no-op unless it is ``"CuTeDSL"``.
        op_name: Optional op identifier classified against
            ``_CUTEDSL_OP_SUPPORTED_SM`` (e.g. ``"triangle_attention"``).
            ``None`` or any unlisted name uses the generic CuTeDSL range.
    """
    if backend_name != "CuTeDSL":
        return
    supported = _cutedsl_supported_sm(op_name)
    if SM_VERSION not in supported:
        pytest.skip(_cutedsl_skip_reason(supported))


def skip_if_not_sm90():
    """Skip when the current GPU isn't Hopper (SM90).

    For tests that assert / exercise an SM90-specific *kernel path* (e.g.
    the Hopper ping-pong dual GEMM) rather than just needing *a* CuTeDSL
    kernel for the op.
    """
    if SM_VERSION != 90:
        pytest.skip(f"requires SM90 (current SM{SM_VERSION})")


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
