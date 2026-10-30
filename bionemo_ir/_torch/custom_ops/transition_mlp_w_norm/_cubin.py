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
"""CUBIN-backed executable adapter for the SM90 transition with its LayerNorm."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_flat,
    tensor_s1_d0,
)

#: The launch ABI this adapter drives: the kernel takes its launch layout (``split``, ``tile_wgs``) as arguments.
LAUNCH_ABI = "transition_mlp_w_norm_sm90_v3"


class TransitionMlpWNormCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Launch one packaged image through the C++ launcher: the image of a shape that takes every launch layout, or its
    persistent one (``persistent``), which takes the full tile at split 1 only.

    The source kernel takes the mask and the item lengths as raw addresses, 0 for none. The launcher checks the spans
    behind them, so this adapter takes the tensors themselves, or ``None``, in their place.

    A library built before launch ABI ``transition_mlp_w_norm_sm90_v3`` launches a kernel without the launch layout
    arguments (and before v2, one without the ``Activation`` axis that requires the mask and the residual). It carries
    no image this adapter can run, so the variant is unavailable there and the op does not ship from it.
    """

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        width: int,
        hidden: int,
        activation: str,
        persistent: bool = False,
    ) -> None:
        activations = getattr(launcher, "Activation", None)
        if activations is None or getattr(launcher, "LAUNCH_ABI", None) != LAUNCH_ABI:
            raise CuTeDSLKernelVariantUnavailable(
                f"this fused transition CUBIN library predates launch ABI {LAUNCH_ABI}"
            )
        try:
            config = launcher.make_kernel_config(
                target_sm,
                launcher.DType.BFLOAT16,
                width,
                hidden,
                activations.SILU_GATE if activation == "silu_gate" else activations.RELU,
                persistent,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No fused transition CUBIN for SM{target_sm}, width={width}, hidden={hidden}, activation={activation}, "
                f"persistent={persistent}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        spec = config.spec
        self.geometry = (int(spec.tile_m), int(spec.num_threads) // 128, int(spec.layouts))
        """Rows per full tile, warpgroups per CTA and the mask of the ``(tile_wgs, split)`` pairs the image takes."""

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        image: torch.Tensor,
        vec: torch.Tensor,
        output: torch.Tensor,
        n_valid: torch.Tensor | None,
        rows: int,
        n: int,
        n_batch: int,
        eps: float,
        residual: int,
        split: int,
        tile_wgs: int,
    ) -> None:
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.x = tensor_flat(library, x)
        if mask is not None:
            params.mask = tensor_flat(library, mask)
        params.image = tensor_flat(library, image)
        params.vec = tensor_s1_d0(library, vec)
        params.output = tensor_flat(library, output)
        if n_valid is not None:
            params.n_valid = tensor_s1_d0(library, n_valid)
        params.rows = rows
        params.n = n
        params.n_batch = n_batch
        params.eps = eps
        params.residual = bool(residual)
        params.split = split
        params.tile_wgs = tile_wgs
        params.stream = current_stream_handle(x)
        self._launcher.launch(self._config, params)
