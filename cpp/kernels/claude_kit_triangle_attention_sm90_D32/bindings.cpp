/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "launcher.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::claude_kit_triangle_attention_sm90_d32
{

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "claude_kit_triangle_attention_sm90_D32",
    "Direct SM90 launcher for the Claude Kit D=32 bfloat16 triangle-attention kernel");

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("q", &LaunchParams::q)
    .def_rw("k", &LaunchParams::k)
    .def_rw("v", &LaunchParams::v)
    .def_rw("bias", &LaunchParams::bias)
    .def_rw("actual_s_kv", &LaunchParams::actual_s_kv)
    .def_rw("output", &LaunchParams::output)
    .def_rw("lse", &LaunchParams::lse)
    .def_rw("softmax_scale", &LaunchParams::softmax_scale)
    .def_rw("i_dim", &LaunchParams::i_dim)
    .def_rw("stream", &LaunchParams::stream);

  module.attr("HEAD_DIM") = 32;
  module.def("attention_dynamic_smem_bytes", &attention_dynamic_smem_bytes);
  module.def(
    "launch",
    [](LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      launch(params);
    },
    "params"_a,
    "Stage the pair bias and launch the attention kernel into preallocated output/LSE buffers.");
}

} // namespace bioir::claude_kit_triangle_attention_sm90_d32
