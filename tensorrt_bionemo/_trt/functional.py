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
from enum import IntEnum
from typing import Callable, Optional, Union

import numpy as np
import tensorrt as trt
from tensorrt_llm_lite._common import default_trtnet
from tensorrt_llm_lite._utils import fp32_array, str_dtype_to_trt
from tensorrt_llm_lite.functional import (Tensor, _add_plugin_info,
                                          _create_tensor, cast, concat,
                                          constant, floordiv, shape)

from .plugin import TRT_BNM_PLUGIN_NAMESPACE


class AttentionBiasType(IntEnum):
    triangle = 0
    pairwise = 1


def identity_sz(s: Tensor,
                z: Tensor,
                use_identity_plugin: bool = True) -> Tensor:
    '''
    Add an identity operation.

    This plugin is used to break Myelin layers. See NVBUGs: NVBug 5300894

    Parameters:
        input : Tensor
            The input tensor.

    Returns:
        The tensor produced by this identity operation.
    '''
    if not use_identity_plugin:
        return s, z
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'IdentitySZ', '1', TRT_BNM_PLUGIN_NAMESPACE)
    assert plg_creator is not None
    pfc = trt.PluginFieldCollection()
    id_plug = plg_creator.create_plugin("identity_sz", pfc)
    plug_inputs = [s.trt_tensor, z.trt_tensor]
    layer = default_trtnet().add_plugin_v2(plug_inputs, id_plug)
    _add_plugin_info(layer, plg_creator, "identity_sz", pfc)

    return _create_tensor(layer.get_output(0), layer), \
        _create_tensor(layer.get_output(1), layer)


def chunk_loop(tensors: list[Tensor],
               chunk_size: int = 0,
               loop_body: Callable = None,
               reshape_output: bool = True,
               name: str = "chunk_loop_output") -> Tensor:
    if chunk_size == 0 or chunk_size is None:
        if loop_body is not None:
            return loop_body(tensors)
        return tensors
    x = tensors[0]
    bs = shape(x, 0)
    bs = cast(bs, trt.int64)
    chunk_size_tensor = constant(np.array(chunk_size, dtype=np.int64))
    niters = cast(floordiv(bs, chunk_size_tensor), trt.int64)

    # Reshape tensors to have the first dimension be the iteration dimension
    reshaped_tensors = []
    for t in tensors:
        s = []
        for i in range(1, t.ndim()):
            s.append(shape(t, i))
        t = t.view(concat([niters, chunk_size_tensor, *s]))
        reshaped_tensors.append(t)
    chunk_loop = default_trtnet().add_loop()
    chunk_loop.add_trip_limit(niters.trt_tensor, trt.TripLimit.COUNT)

    data = []
    for t in reshaped_tensors:
        # For each tensor, we need to add a loop iterator
        iterator = chunk_loop.add_iterator(t.trt_tensor, 0, False)
        data.append(_create_tensor(iterator.get_output(0), iterator))

    if loop_body is not None:
        loop_body_output = loop_body(data)
    else:
        assert len(data) == 1
        loop_body_output = data[0]

    trt_output_layer = chunk_loop.add_loop_output(loop_body_output.trt_tensor,
                                                  trt.LoopOutput.CONCATENATE,
                                                  0)
    trt_output_layer.name = name
    trt_output_layer.set_input(1, niters.trt_tensor)
    output = _create_tensor(trt_output_layer.get_output(0), trt_output_layer)
    if reshape_output:
        s = []
        for i in range(2, output.ndim()):
            s.append(shape(output, i))
        output = output.view(concat([bs, *s]))
    return output


class AttentionBackend(IntEnum):
    TRIFAST = 0
    CUEQUIV = 1

    @staticmethod
    def from_str(backend: str) -> "AttentionBackend":
        if backend == "TRIFAST":
            return AttentionBackend.TRIFAST
        elif backend == "CUEQUIV":
            return AttentionBackend.CUEQUIV
        else:
            raise ValueError(f"Invalid attention backend: {backend}")


def triangle_attention(q: Tensor,
                       k: Tensor,
                       v: Tensor,
                       bias: Tensor,
                       mask: Optional[Tensor],
                       num_heads: int,
                       head_dim: int,
                       dtype: str = "float32",
                       backend: AttentionBackend = AttentionBackend.CUEQUIV,
                       use_tf32: bool = False) -> tuple[Tensor, Tensor]:
    assert backend is not None, "Attention backend must be specified"
    if isinstance(backend, str):
        backend = AttentionBackend.from_str(backend)
    dtype = "float32" if dtype is None else dtype
    tri_attn_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'TriAttn', '1', TRT_BNM_PLUGIN_NAMESPACE)
    assert tri_attn_plg_creator is not None
    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32),
                             trt.PluginFieldType.INT32)
    head_dim = trt.PluginField("head_dim", np.array(head_dim, dtype=np.int32),
                               trt.PluginFieldType.INT32)
    backend = trt.PluginField("backend", np.array(backend.value,
                                                  dtype=np.int32),
                              trt.PluginFieldType.INT32)
    use_tf32 = trt.PluginField("use_tf32", np.array(use_tf32, dtype=np.bool_),
                               trt.PluginFieldType.INT8)
    if isinstance(dtype, str):
        type_id = int(str_dtype_to_trt(dtype))
    else:
        type_id = int(dtype)
    pf_type = trt.PluginField("type_id", np.array([type_id], np.int32),
                              trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection(
        [nheads, head_dim, backend, use_tf32, pf_type])

    tri_attn_plug = tri_attn_plg_creator.create_plugin("tri_attn", pfc)
    plug_inputs = [q.trt_tensor, k.trt_tensor, v.trt_tensor, bias.trt_tensor]
    if mask is not None:
        plug_inputs += [mask.trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, tri_attn_plug)
    _add_plugin_info(layer, tri_attn_plg_creator, "tri_attn", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    lse = _create_tensor(layer.get_output(1), layer)
    return output, lse


def dynamic_const_tensor(
        shape: Tensor,
        value: float = 0.0,
        dtype: Union[str, trt.DataType] = 'float32') -> Tensor:
    """ Create a dynamic tensor of zeros. """
    low = constant(fp32_array(float(0.0)))
    high = constant(fp32_array([float(0.0)] * shape.shape[0]))

    layer = default_trtnet().add_fill([0], trt.FillOperation.LINSPACE,
                                      trt.float32)

    layer.set_input(0, shape.trt_tensor)
    layer.set_input(1, low.trt_tensor)
    layer.set_input(2, high.trt_tensor)

    if value != 0.0:
        c = constant(fp32_array(value))
        return cast(_create_tensor(layer.get_output(0), layer) + c, dtype)
    return cast(_create_tensor(layer.get_output(0), layer), dtype)
