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
"""SM90 BF16 D=32 triangle attention adapted from the Uplifting Biomolecular Modeling M1 kernel."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import cast

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    tensor_s1_d0,
    tensor_s3_d2,
    tensor_s3_d2_static,
    tensor_s4_d3,
)

from ..interface import AttentionBackend
from .cutedsl import (
    TriangleAttentionCuTeLeftMask,
    TriangleAttentionCuTeLeftMaskMetadata,
    _TriangleAttentionVariant,
)

_KERNEL_LIBRARY_MODULE = "bionemo_ir.libs._cutedsl_kernels"
_KERNEL_SUBMODULE = "claude_kit_triangle_attention_sm90_D32"


class ClaudeKitTriangleAttentionUnavailable(RuntimeError):
    """The native Claude-kit triangle-attention launcher is unavailable."""


class ClaudeKitTriangleAttentionMetadata(TriangleAttentionCuTeLeftMaskMetadata):
    """Metadata shared with the CuTeDSL left-mask backend."""


def _load_claude_kit_modules() -> tuple[ModuleType, ModuleType]:
    """Load the native extension and Claude-kit submodule on first use."""
    try:
        library = importlib.import_module(_KERNEL_LIBRARY_MODULE)
    except (ImportError, OSError) as error:
        raise ClaudeKitTriangleAttentionUnavailable(
            f"Cannot import {_KERNEL_LIBRARY_MODULE!r}; rebuild BioIR with the Claude-kit CUDA kernel"
        ) from error

    launcher = getattr(library, _KERNEL_SUBMODULE, None)
    if launcher is None:
        raise ClaudeKitTriangleAttentionUnavailable(
            f"{_KERNEL_LIBRARY_MODULE!r} has no {_KERNEL_SUBMODULE!r} submodule; "
            "rebuild BioIR with the Claude-kit CUDA kernel"
        )
    return library, cast(ModuleType, launcher)


class _ClaudeKitExecutable(CuTeDSLKernelLibraryExecutable):
    """Adapt the native launcher to the compiled-kernel call ABI; an empty ``lse`` skips that store."""

    def __init__(self, kernel_library: ModuleType, launcher: ModuleType):
        self._kernel_library = kernel_library
        self._launcher = launcher

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        actual_s_kv: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        _softmax_scale_log2: float,
        softmax_scale: float,
        i_dim: int,
    ) -> None:
        params = self._launcher.LaunchParams()
        params.q = tensor_s3_d2_static(self._kernel_library, q)
        params.k = tensor_s3_d2_static(self._kernel_library, k)
        params.v = tensor_s3_d2_static(self._kernel_library, v)
        params.actual_s_kv = tensor_s1_d0(self._kernel_library, actual_s_kv)
        params.bias = tensor_s4_d3(self._kernel_library, bias)
        params.output = tensor_s3_d2_static(self._kernel_library, output)
        params.lse = tensor_s3_d2(self._kernel_library, lse)
        params.softmax_scale = softmax_scale
        params.i_dim = i_dim
        params.stream = torch.cuda.current_stream(q.device).cuda_stream
        self._launcher.launch(params)


class ClaudeKitTriangleAttentionSM90D32(TriangleAttentionCuTeLeftMask):
    """Direct SM90 BF16 D=32 left-mask triangle-attention backend.

    Same contract as :class:`TriangleAttentionCuTeLeftMask`; packed Q/K/V
    views are read in place.

    Args:
        layer_idx: The index of the attention layer.
        num_heads: The number of attention heads.
        head_dim: The dimension of each attention head.
        num_kv_heads: The number of key-value heads; must equal ``num_heads``.
    """

    Metadata = ClaudeKitTriangleAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
    ):
        # Skip the CuTeDSL initializer, which queries CUDA; the first forward validates SM90.
        AttentionBackend.__init__(self, layer_idx, num_heads, head_dim, num_kv_heads)
        if self.num_heads != self.num_kv_heads:
            raise ValueError("num_heads must be equal to num_kv_heads")
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        self._sm_version = capability[0] * 10 + capability[1]
        self._last_executable: _ClaudeKitExecutable | None = None
        self._last_variant: _TriangleAttentionVariant | None = None
        self._last_force_cubin: bool | None = None
        self._last_variant_slot: tuple[tuple, _TriangleAttentionVariant] | None = None

    @staticmethod
    def supports(sm_version: int, dtype: torch.dtype, head_dim: int) -> bool:
        """Return whether the native kernel supports this static variant."""
        return sm_version == 90 and dtype == torch.bfloat16 and head_dim == 32

    @staticmethod
    def native_available() -> bool:
        """Return whether the lazily loaded extension exports this launcher."""
        try:
            _load_claude_kit_modules()
        except ClaudeKitTriangleAttentionUnavailable:
            return False
        return True

    def _get_executable(
        self,
        variant: _TriangleAttentionVariant,
        ct_dtype: object,
        align_elems: int,
        sm_scale: float,
        i_dim: int,
    ) -> _ClaudeKitExecutable:
        del ct_dtype, align_elems, sm_scale, i_dim
        if not self.supports(self._sm_version, variant.dtype, self.head_dim):
            raise ValueError(
                "ClaudeKit triangle attention requires SM90, bfloat16, and head_dim=32; "
                f"got SM{self._sm_version}, dtype={variant.dtype}, head_dim={self.head_dim}"
            )
        library, launcher = _load_claude_kit_modules()
        return _ClaudeKitExecutable(library, launcher)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: ClaudeKitTriangleAttentionMetadata | None = None,
        output: torch.Tensor | None = None,
        output_lse: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the native Claude-kit left-mask kernel.

        Args:
            q: ``[B, I, J, H*D]`` or ``[B, I, J, H, D]`` BF16 CUDA tensor.
            k: Tensor with the same shape, dtype, and device as ``q``.
            v: Tensor with the same shape, dtype, and device as ``q``.
            biases: ``[actual_s_kv, pair_bias]``. Lengths accept ``[B, I]``,
                ``[B*I]``, or ``[B]``; pair bias is ``[B, H, J, J]``.
            metadata: Optional packed-QKV metadata.
            output: Optional preallocated ``[B*I, J, H, 32]`` BF16 buffer.
            output_lse: Optional preallocated ``[B*I, J, H, 1]`` FP32 buffer.
            **kwargs: Ignored compatibility arguments.

        Returns:
            Triangle-attention output with shape ``[B, I, J, H, 32]``.
        """
        if not q.is_cuda:
            raise ValueError("ClaudeKit triangle attention requires CUDA tensors")
        if not self.supports(self._sm_version, q.dtype, self.head_dim):
            raise ValueError(
                "ClaudeKit triangle attention requires SM90, bfloat16, and head_dim=32; "
                f"got SM{self._sm_version}, dtype={q.dtype}, head_dim={self.head_dim}"
            )
        if biases is None or len(biases) < 2:
            raise ValueError("ClaudeKit triangle attention expects biases=[actual_s_kv, pair_bias]")
        return super().forward(
            q,
            k,
            v,
            biases=biases,
            metadata=metadata,
            output=output,
            output_lse=output_lse,
            **kwargs,
        )
