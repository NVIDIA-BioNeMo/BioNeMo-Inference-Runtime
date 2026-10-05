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

"""Inference kernels for EDM integration."""

from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import TritonKernelCache


@triton.jit(do_not_specialize=["count", "scale"], do_not_specialize_on_alignment=["sigma_last_ptr", "sigma_hat_ptr"])
def _churn(x_ptr, noise_ptr, sigma_last_ptr, sigma_hat_ptr, output_ptr, count, scale, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    x = tl.load(x_ptr + index, valid, 0)
    noise = tl.load(noise_ptr + index, valid, 0)
    sigma_last = tl.load(sigma_last_ptr)
    sigma_hat = tl.load(sigma_hat_ptr)
    deviation = tl.sqrt_rn(sigma_hat * sigma_hat - sigma_last * sigma_last)
    tl.store(output_ptr + index, x + (scale * deviation) * noise, valid)


class _ChurnKernel(TritonKernelCache):
    def __init__(self) -> None:
        self.kernel = self.compile_for_dtypes(
            _churn,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty((), device="cuda", dtype=dtype),
                torch.empty((), device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                1,
                1.5,
            ),
            grid=(0,),
            BLOCK=256,
            enable_fp_fusion=False,
        )[torch.float32]


@cache
def _churn_kernel(device: int) -> _ChurnKernel:
    with torch.cuda.device(device):
        return _ChurnKernel()


def churn_update(
    x: torch.Tensor, noise: torch.Tensor, sigma_last: torch.Tensor, sigma_hat: torch.Tensor, scale: float
) -> torch.Tensor:
    """Update contiguous CUDA FP32 tensors accepted by ``supports_churn``.

    Args:
        x: Coordinate tensor of any shape.
        noise: Standard normal noise matching ``x``.
        sigma_last: Scalar previous noise level on the same device.
        sigma_hat: Scalar churn noise level on the same device.
        scale: Noise standard-deviation multiplier.
    """
    output = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            kernel = _churn_kernel(x.device.index).kernel
            grid = triton.cdiv(x.numel(), 256)
            driver = kernel.driver
            if driver is not None:
                values = (
                    x.data_ptr(),
                    noise.data_ptr(),
                    sigma_last.data_ptr(),
                    sigma_hat.data_ptr(),
                    output.data_ptr(),
                    x.numel(),
                    scale,
                )
                driver.launch_with(values, grid)
            else:
                kernel.launch((grid,), x, noise, sigma_last, sigma_hat, output, x.numel(), scale, 256)
    return output


def supports_churn(x: torch.Tensor, y: torch.Tensor, sigma_last: torch.Tensor, sigma_hat: torch.Tensor) -> bool:
    """Check whether churn inputs support inference CUBIN dispatch.

    Args:
        x: Coordinate tensor.
        y: Standard normal noise matching ``x``.
        sigma_last: Scalar previous noise level.
        sigma_hat: Scalar churn noise level.
    """
    tensors = (x, y, sigma_last, sigma_hat)
    return (
        x.is_cuda
        and not torch.compiler.is_compiling()
        and x.shape == y.shape
        and sigma_hat.ndim == 0
        and sigma_last.ndim == 0
        and x.numel() < 2**31
        and all(t.dtype == torch.float32 and t.device == x.device and not t.is_neg() for t in tensors)
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (x, y))
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    )
