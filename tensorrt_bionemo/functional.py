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
from tensorrt_llm._common import default_trtnet
from tensorrt_llm.functional import (Tensor, _create_tensor, cast, concat,
                                     constant, floordiv, shape)


class AttentionBiasType(IntEnum):
    triangle = 0
    pairwise = 1


def chunk_loop(x: Tensor,
               chunk_size: int = 0,
               loop_body: Callable = None,
               reshape_output: bool = True,
               name: str = "chunk_loop_output") -> Tensor:
    if chunk_size == 0 or chunk_size is None:
        return x
    bs = shape(x, 0)
    bs = cast(bs, trt.int64)
    chunk_size_tensor = constant(np.array(chunk_size, dtype=np.int64))
    niters = cast(floordiv(bs, chunk_size_tensor), trt.int64)
    s = []
    for i in range(1, x.ndim()):
        s.append(shape(x, i))
    x = x.view(concat([niters, chunk_size_tensor, *s]))
    chunk_loop = default_trtnet().add_loop()
    chunk_loop.add_trip_limit(niters.trt_tensor, trt.TripLimit.COUNT)

    iterator = chunk_loop.add_iterator(x.trt_tensor, 0, False)
    data = _create_tensor(iterator.get_output(0), iterator)
    if loop_body is not None:
        data = loop_body(data)

    trt_output_layer = chunk_loop.add_loop_output(data.trt_tensor,
                                                  trt.LoopOutput.CONCATENATE, 0)
    trt_output_layer.name = name
    trt_output_layer.set_input(1, niters.trt_tensor)
    output = _create_tensor(trt_output_layer.get_output(0), trt_output_layer)
    if reshape_output:
        output = output.view(concat([bs, *s]))
    return output
