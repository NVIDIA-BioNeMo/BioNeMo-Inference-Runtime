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
from typing import Callable

import numpy as np
import tensorrt as trt
from tensorrt_llm._common import default_net, default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import (Tensor, _add_plugin_info, _create_tensor,
                                     cast, concat, constant, floordiv, shape)

from .plugin import TRT_BNM_PLUGIN_NAMESPACE


class AttentionBiasType(IntEnum):
    triangle = 0
    pairwise = 1


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
                                                  trt.LoopOutput.CONCATENATE, 0)
    trt_output_layer.name = name
    trt_output_layer.set_input(1, niters.trt_tensor)
    output = _create_tensor(trt_output_layer.get_output(0), trt_output_layer)
    if reshape_output:
        s = []
        for i in range(2, output.ndim()):
            s.append(shape(output, i))
        output = output.view(concat([bs, *s]))
    return output


def send_recv(send_tensor: Tensor,
              src: int,
              tgt: int,
              group: list[int],
              group_stride: int = 1) -> Tensor:
    '''
    Add an operation that performs a send from a rank to another and a recv from another rank to a rank, simunestously.
    Parameters:
        send_tensor (Tensor): The tensor to send.
        src (int): The source rank.
        tgt (int): The target rank.
        group (List[int]): The group of ranks.
        group_stride (int): The stride of the group.
    Returns:
        The received tensor.
    '''
    send_recv_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'SendRecv', '1', TRT_BNM_PLUGIN_NAMESPACE)
    assert send_recv_plg_creator is not None

    src = trt.PluginField("src_rank", np.array(src, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    tgt = trt.PluginField("tgt_rank", np.array(tgt, dtype=np.int32),
                          trt.PluginFieldType.INT32)
    group = trt.PluginField("group", np.array(group, dtype=np.int32),
                            trt.PluginFieldType.INT32)
    group_stride = trt.PluginField("group_stride",
                                   np.array(group_stride, dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    p_dtype = default_net().plugin_config.nccl_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32),
        trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([src, tgt, group, group_stride, pf_type])
    send_recv_plug = send_recv_plg_creator.create_plugin("send_recv", pfc)
    plug_inputs = [send_tensor.cast(p_dtype).trt_tensor]

    layer = default_trtnet().add_plugin_v2(plug_inputs, send_recv_plug)
    _add_plugin_info(layer, send_recv_plg_creator, "send_recv", pfc)
    return _create_tensor(layer.get_output(0), layer).cast(send_tensor.dtype)
