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
"""Single-kernel LayerNorm without learnable affine parameters (γ/β).

Used by the mega-GEMM precomputed bias path in OpenFold3DiffusionTransformer
where per-layer LN γ is fused into the projection weight matrix, leaving only
the normalization step: ``(x - mean) / sqrt(var + eps)``.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_no_affine_kernel(
    Z_ptr,
    OUT_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
    OUT_DTYPE: tl.constexpr = tl.bfloat16,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    z = tl.load(Z_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(z, axis=0) / D
    z_c = z - mean
    var = tl.sum(z_c * z_c, axis=0) / D
    z_hat = z_c * tl.rsqrt(var + eps)

    tl.store(OUT_ptr + row * D + offs, z_hat.to(OUT_DTYPE), mask=mask)


_TORCH_TO_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


def fused_layer_norm_no_affine(z: torch.Tensor,
                               eps: float = 1e-5) -> torch.Tensor:
    """Single-kernel LayerNorm without γ/β.

    Reads input → computes stats in f32 → normalizes → writes back in the
    original dtype, replacing the multi-kernel PyTorch sequence.

    Supported dtypes: ``bfloat16``, ``float16``, ``float32``.

    Args:
        z: Input tensor (contiguous, any shape). Last dim is normalized.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor with same shape and dtype as input.
    """
    tl_dtype = _TORCH_TO_TL_DTYPE.get(z.dtype)
    if tl_dtype is None:
        raise ValueError(f"Unsupported dtype {z.dtype}; expected one of "
                         f"{list(_TORCH_TO_TL_DTYPE.keys())}")

    z = z.contiguous()
    D = z.shape[-1]
    flat = z.reshape(-1, D)
    N_rows = flat.shape[0]
    out = torch.empty_like(flat)

    BLOCK_D = triton.next_power_of_2(D)
    _layer_norm_no_affine_kernel[(N_rows, )](
        flat,
        out,
        D=D,
        BLOCK_D=BLOCK_D,
        eps=eps,
        OUT_DTYPE=tl_dtype,
        num_warps=min(8, max(1, BLOCK_D // 32)),
    )
    return out.reshape(z.shape)
