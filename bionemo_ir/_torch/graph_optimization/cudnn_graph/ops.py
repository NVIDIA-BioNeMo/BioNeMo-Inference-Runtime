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
"""Reusable eager-callable cuDNN operation graphs."""

from __future__ import annotations

import threading
from collections.abc import Callable

import cudnn
import torch

from .cache import CudnnGraphCache
from .dynamic import DynamicGraph

type LinearGraphSignature = tuple[
    torch.device,
    torch.dtype,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]

_DYNAMIC_CACHE_ROWS = 64 * 64
# Small-channel plans can reject otherwise valid row overrides and select a
# different BF16 accumulation path. Supported production widths are >= 256.
_DYNAMIC_LINEAR_MIN_DIM = 128
_KERNEL_CACHE_LOCK = threading.Lock()
_LINEAR_RELU_KERNEL_CACHES: dict[tuple[torch.device, torch.dtype], object] = {}
_LINEAR_MASK_KERNEL_CACHES: dict[tuple[torch.device, torch.dtype], object] = {}


def _cudnn_data_type(dtype: torch.dtype) -> cudnn.data_type:
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    raise ValueError(f"cuDNN operation graphs do not support output dtype {dtype}")


def _kernel_cache(
    caches: dict[tuple[torch.device, torch.dtype], object],
    device: torch.device,
    dtype: torch.dtype,
) -> object:
    key = (device, dtype)
    with _KERNEL_CACHE_LOCK:
        if key not in caches:
            with torch.cuda.device(device):
                caches[key] = cudnn.create_kernel_cache()
        return caches[key]


def _prepared_linear_signature(
    device: torch.device,
    dtype: torch.dtype,
    input_dim: int,
    output_dim: int,
) -> LinearGraphSignature:
    return (
        device,
        dtype,
        input_dim,
        output_dim,
        (input_dim * output_dim, 1, input_dim),
        (output_dim, output_dim, 1),
    )


def _indexed_cuda_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


type _DynamicLinearEpilogueMaker = Callable[
    [DynamicGraph, cudnn.tensor],
    tuple[cudnn.tensor, tuple[cudnn.tensor, ...]],
]


def _make_dynamic_relu_epilogue(
    dynamic_graph: DynamicGraph,
    output: cudnn.tensor,
) -> tuple[cudnn.tensor, tuple[cudnn.tensor, ...]]:
    output = dynamic_graph.graph.relu(
        output,
        compute_data_type=cudnn.data_type.FLOAT,
        name="relu",
    )
    return output, ()


def _make_dynamic_mask_epilogue(
    dynamic_graph: DynamicGraph,
    output: cudnn.tensor,
) -> tuple[cudnn.tensor, tuple[cudnn.tensor, ...]]:
    mask = dynamic_graph.tensor(
        (1, _DYNAMIC_CACHE_ROWS, 1),
        (_DYNAMIC_CACHE_ROWS, 1, 1),
        "mask",
    )
    output = dynamic_graph.graph.mul(
        output,
        mask,
        compute_data_type=cudnn.data_type.FLOAT,
        name="mask",
    )
    return output, (mask,)


class _DynamicLinearEpiPlan:
    def __init__(
        self,
        signature: LinearGraphSignature,
        kernel_caches: dict[tuple[torch.device, torch.dtype], object],
        make_epilogue: _DynamicLinearEpilogueMaker,
    ) -> None:
        device, dtype, input_dim, output_dim, weight_stride, bias_stride = signature
        self.input_dim = input_dim
        self.output_dim = output_dim
        dynamic_graph = DynamicGraph(
            device,
            dtype,
            _kernel_cache(kernel_caches, device, dtype),
        )
        graph = dynamic_graph.graph
        self.x = dynamic_graph.tensor(
            (1, _DYNAMIC_CACHE_ROWS, input_dim),
            (_DYNAMIC_CACHE_ROWS * input_dim, input_dim, 1),
            "x",
        )
        self.weight = dynamic_graph.tensor((1, input_dim, output_dim), weight_stride, "weight")
        self.bias = dynamic_graph.tensor((1, 1, output_dim), bias_stride, "bias")
        output = graph.matmul(
            self.x,
            self.weight,
            compute_data_type=cudnn.data_type.FLOAT,
            name="linear",
        )
        output = graph.bias(
            output,
            self.bias,
            compute_data_type=cudnn.data_type.FLOAT,
            name="bias",
        )
        output.set_data_type(_cudnn_data_type(dtype))
        self.output, self._epilogue_inputs = make_epilogue(dynamic_graph, output)
        self.output.set_output(True).set_data_type(_cudnn_data_type(dtype))
        dynamic_graph.build()
        self._graph = dynamic_graph
        self._override_uids = tuple(tensor.get_uid() for tensor in (self.x, *self._epilogue_inputs, self.output))
        self._unsupported_rows: set[int] = set()

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *epilogue_inputs: torch.Tensor,
    ) -> torch.Tensor | None:
        if len(epilogue_inputs) != len(self._epilogue_inputs):
            raise ValueError(f"expected {len(self._epilogue_inputs)} epilogue inputs, got {len(epilogue_inputs)}")
        rows = x.shape[1]
        if rows in self._unsupported_rows:
            return None
        output = torch.empty((1, rows, self.output_dim), device=x.device, dtype=x.dtype)
        bindings = {
            self.x: x,
            self.weight: weight,
            self.bias: bias,
            self.output: output,
        }
        bindings.update(zip(self._epilogue_inputs, epilogue_inputs, strict=True))
        dynamic_tensors = (x, *epilogue_inputs, output)
        executed = self._graph.execute(
            bindings,
            self._override_uids,
            tuple(tuple(tensor.shape) for tensor in dynamic_tensors),
            tuple(tuple(tensor.stride()) for tensor in dynamic_tensors),
        )
        if executed:
            return output
        self._unsupported_rows.add(rows)
        return None


type _LinearEpilogueMaker = Callable[
    [cudnn.Graph, cudnn.tensor, tuple[torch.Tensor, ...]],
    cudnn.tensor,
]


def _make_linear_relu_epilogue(
    graph: cudnn.Graph,
    output: cudnn.tensor,
    _epilogue_inputs: tuple[torch.Tensor, ...],
) -> cudnn.tensor:
    return graph.relu(
        output,
        compute_data_type=cudnn.data_type.FLOAT,
        name="relu",
    )


def _make_linear_mask_epilogue(
    graph: cudnn.Graph,
    output: cudnn.tensor,
    epilogue_inputs: tuple[torch.Tensor, ...],
) -> cudnn.tensor:
    (mask,) = epilogue_inputs
    return graph.mul(
        output,
        mask,
        compute_data_type=cudnn.data_type.FLOAT,
        name="mask",
    )


class _LinearEpiPlan:
    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        epilogue_inputs: tuple[torch.Tensor, ...],
        make_epilogue: _LinearEpilogueMaker,
    ) -> None:
        output_type = _cudnn_data_type(x.dtype)
        with torch.cuda.device(x.device):
            with cudnn.Graph(
                handle="auto",
                io_data_type=x.dtype,
                intermediate_data_type=torch.float32,
                compute_data_type=torch.float32,
            ) as graph:
                output = graph.matmul(
                    x,
                    weight,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="linear",
                )
                output = graph.bias(
                    output,
                    bias,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="bias",
                )
                output.set_data_type(output_type)
                output = make_epilogue(graph, output, epilogue_inputs)
                output.set_output(True).set_data_type(output_type)
            graph.set_io_tuples([x, weight, bias, *epilogue_inputs], [output])
        self._graph = graph

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *epilogue_inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self._graph(x, weight, bias, *epilogue_inputs)


class _ScaleShiftMaskPlan:
    def __init__(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        output_type = _cudnn_data_type(value.dtype)
        with torch.cuda.device(value.device):
            with cudnn.Graph(
                handle="auto",
                io_data_type=value.dtype,
                intermediate_data_type=torch.float32,
                compute_data_type=torch.float32,
            ) as graph:
                output = graph.mul(
                    value,
                    scale,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="scale",
                )
                output.set_data_type(output_type)
                output = graph.add(
                    output,
                    shift,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="shift",
                )
                output.set_data_type(output_type)
                output = graph.mul(
                    output,
                    mask,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="mask",
                )
                output.set_output(True).set_data_type(output_type)
            graph.set_io_tuples([value, scale, shift, mask], [output])
        self._graph = graph

    def __call__(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._graph(value, scale, shift, mask)


class _AddAddMaskPlan:
    def __init__(
        self,
        value: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        output_type = _cudnn_data_type(value.dtype)
        with torch.cuda.device(value.device):
            with cudnn.Graph(
                handle="auto",
                io_data_type=value.dtype,
                intermediate_data_type=torch.float32,
                compute_data_type=torch.float32,
            ) as graph:
                output = graph.add(
                    value,
                    first,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="add_first",
                )
                output.set_data_type(output_type)
                output = graph.add(
                    output,
                    second,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="add_second",
                )
                output.set_data_type(output_type)
                output = graph.mul(
                    output,
                    mask,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="mask",
                )
                output.set_output(True).set_data_type(output_type)
            graph.set_io_tuples([value, first, second, mask], [output])
        self._graph = graph

    def __call__(
        self,
        value: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._graph(value, first, second, mask)


class _AddMaskPlan:
    def __init__(self, value: torch.Tensor, update: torch.Tensor, mask: torch.Tensor) -> None:
        output_type = _cudnn_data_type(value.dtype)
        with torch.cuda.device(value.device):
            with cudnn.Graph(
                handle="auto",
                io_data_type=value.dtype,
                intermediate_data_type=torch.float32,
                compute_data_type=torch.float32,
            ) as graph:
                output = graph.add(
                    value,
                    update,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="add",
                )
                output.set_data_type(output_type)
                output = graph.mul(
                    output,
                    mask,
                    compute_data_type=cudnn.data_type.FLOAT,
                    name="mask",
                )
                output.set_output(True).set_data_type(output_type)
            graph.set_io_tuples([value, update, mask], [output])
        self._graph = graph

    def __call__(self, value: torch.Tensor, update: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._graph(value, update, mask)


_LINEAR_RELU_CACHE = CudnnGraphCache[_LinearEpiPlan]("linear + bias + ReLU")
_LINEAR_MASK_CACHE = CudnnGraphCache[_LinearEpiPlan]("linear + bias + mask")
_DYNAMIC_LINEAR_RELU_CACHE = CudnnGraphCache[_DynamicLinearEpiPlan]("dynamic linear + bias + ReLU")
_DYNAMIC_LINEAR_MASK_CACHE = CudnnGraphCache[_DynamicLinearEpiPlan]("dynamic linear + bias + mask")
# cuDNN 9.22's generic pointwise fusion engines reject dynamic-shape plans for
# these topologies. Keep exact-layout plans until the backend supports them.
_SCALE_SHIFT_MASK_CACHE = CudnnGraphCache[_ScaleShiftMaskPlan]("scale + shift + mask")
_ADD_ADD_MASK_CACHE = CudnnGraphCache[_AddAddMaskPlan]("two additions + mask")
_ADD_MASK_CACHE = CudnnGraphCache[_AddMaskPlan]("addition + mask")


def _dynamic_linear_relu_plan(signature: LinearGraphSignature) -> _DynamicLinearEpiPlan | None:
    return _DYNAMIC_LINEAR_RELU_CACHE.get_or_create_signature(
        signature,
        lambda: _DynamicLinearEpiPlan(
            signature,
            _LINEAR_RELU_KERNEL_CACHES,
            _make_dynamic_relu_epilogue,
        ),
    )


def _dynamic_linear_mask_plan(signature: LinearGraphSignature) -> _DynamicLinearEpiPlan | None:
    return _DYNAMIC_LINEAR_MASK_CACHE.get_or_create_signature(
        signature,
        lambda: _DynamicLinearEpiPlan(
            signature,
            _LINEAR_MASK_KERNEL_CACHES,
            _make_dynamic_mask_epilogue,
        ),
    )


def prepare_cudnn_linear_relu(
    device: torch.device,
    dtype: torch.dtype,
    input_dim: int,
    output_dim: int,
) -> _DynamicLinearEpiPlan | None:
    """Build one dynamic linear, bias, and ReLU plan for every row count.

    Prefer :func:`cudnn_linear_relu`, whose static plan is faster at every row
    count the released models use. Hold this plan only to cover row counts a
    caller cannot enumerate.

    Args:
        device: CUDA device on which the plan will execute.
        dtype: Input, weight, bias, and output dtype.
        input_dim: GEMM reduction dimension.
        output_dim: GEMM output dimension.

    Returns:
        The prepared plan, or ``None`` when the dimensions are unsupported.
    """
    if device.type != "cuda":
        return None
    if min(input_dim, output_dim) < _DYNAMIC_LINEAR_MIN_DIM:
        return None
    device = _indexed_cuda_device(device)
    signature = _prepared_linear_signature(device, dtype, input_dim, output_dim)
    return _dynamic_linear_relu_plan(signature)


def prepare_cudnn_linear_mask(
    device: torch.device,
    dtype: torch.dtype,
    input_dim: int,
    output_dim: int,
) -> _DynamicLinearEpiPlan | None:
    """Build one dynamic linear, bias, and mask plan for every row count.

    Prefer :func:`cudnn_linear_mask`, whose static plan is faster at every row
    count the released models use. Hold this plan only to cover row counts a
    caller cannot enumerate.

    Args:
        device: CUDA device on which the plan will execute.
        dtype: Input, weight, bias, mask, and output dtype.
        input_dim: GEMM reduction dimension.
        output_dim: GEMM output dimension.

    Returns:
        The prepared plan, or ``None`` when the dimensions are unsupported.
    """
    if device.type != "cuda":
        return None
    if min(input_dim, output_dim) < _DYNAMIC_LINEAR_MIN_DIM:
        return None
    device = _indexed_cuda_device(device)
    signature = _prepared_linear_signature(device, dtype, input_dim, output_dim)
    return _dynamic_linear_mask_plan(signature)


def cudnn_linear_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    """Apply matmul, bias, and ReLU through one static cuDNN graph.

    Args:
        x: Input with shape ``[..., M, K]``.
        weight: Weight with shape ``[..., K, N]``.
        bias: Bias broadcastable to ``[..., M, N]``.

    Returns:
        The fused output, or ``None`` when cuDNN rejects the tensor signature.
    """
    inputs = (x, weight, bias)
    plan = _LINEAR_RELU_CACHE.get_or_create(
        inputs,
        lambda: _LinearEpiPlan(x, weight, bias, (), _make_linear_relu_epilogue),
    )
    return None if plan is None else plan(*inputs)


def cudnn_linear_mask(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Apply matmul, bias, and mask through one static cuDNN graph.

    Args:
        x: Input with shape ``[..., M, K]``.
        weight: Weight with shape ``[..., K, N]``.
        bias: Bias broadcastable to ``[..., M, N]``.
        mask: Multiplicative mask broadcastable to ``[..., M, N]``.

    Returns:
        The fused output, or ``None`` when cuDNN rejects the tensor signature.
    """
    inputs = (x, weight, bias, mask)
    plan = _LINEAR_MASK_CACHE.get_or_create(
        inputs,
        lambda: _LinearEpiPlan(x, weight, bias, (mask,), _make_linear_mask_epilogue),
    )
    return None if plan is None else plan(*inputs)


def cudnn_scale_shift_mask(
    value: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Apply ``(value * scale + shift) * mask`` through one cached graph.

    Args:
        value: Input tensor.
        scale: Multiplicative tensor broadcastable to ``value``.
        shift: Additive tensor broadcastable to ``value``.
        mask: Final multiplicative tensor broadcastable to ``value``.

    Returns:
        The fused output, or ``None`` when cuDNN rejects the tensor signature.
    """
    inputs = (value, scale, shift, mask)
    plan = _SCALE_SHIFT_MASK_CACHE.get_or_create(inputs, lambda: _ScaleShiftMaskPlan(*inputs))
    return None if plan is None else plan(*inputs)


def cudnn_add_add_mask(
    value: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Apply ``(value + first + second) * mask`` through one cached graph.

    Args:
        value: Input tensor.
        first: First additive tensor broadcastable to ``value``.
        second: Second additive tensor broadcastable to ``value``.
        mask: Final multiplicative tensor broadcastable to ``value``.

    Returns:
        The fused output, or ``None`` when cuDNN rejects the tensor signature.
    """
    inputs = (value, first, second, mask)
    plan = _ADD_ADD_MASK_CACHE.get_or_create(inputs, lambda: _AddAddMaskPlan(*inputs))
    return None if plan is None else plan(*inputs)


def cudnn_add_mask(
    value: torch.Tensor,
    update: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Apply ``(value + update) * mask`` through one cached cuDNN graph.

    Args:
        value: Input tensor.
        update: Additive tensor broadcastable to ``value``.
        mask: Final multiplicative tensor broadcastable to ``value``.

    Returns:
        The fused output, or ``None`` when cuDNN rejects the tensor signature.
    """
    inputs = (value, update, mask)
    plan = _ADD_MASK_CACHE.get_or_create(inputs, lambda: _AddMaskPlan(*inputs))
    return None if plan is None else plan(*inputs)
