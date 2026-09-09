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

#include "pair_weighted_averaging_registry.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::pair_weighted_averaging
{

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "pair_weighted_averaging", "Direct CUDA Driver launcher for precompiled CuTeDSL pair-weighted-averaging CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("target_sm", &KernelConfig::target_sm)
    .def_ro("D", &KernelConfig::D)
    .def_ro("c_m", &KernelConfig::c_m)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("n_anchor", &KernelConfig::n_anchor)
    .def_ro("s_anchor", &KernelConfig::s_anchor)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; })
    .def_prop_ro("tile_i", [](KernelConfig const& config) { return config.embedded_image->tile_i; })
    .def_prop_ro("tile_s", [](KernelConfig const& config) { return config.embedded_image->tile_s; })
    .def_prop_ro("tile_j", [](KernelConfig const& config) { return config.embedded_image->tile_j; })
    .def_prop_ro("num_threads", [](KernelConfig const& config) { return config.embedded_image->num_threads; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("w", &LaunchParams::w)
    .def_rw("v", &LaunchParams::v)
    .def_rw("g", &LaunchParams::g)
    .def_rw("weight", &LaunchParams::weight)
    .def_rw("output", &LaunchParams::output)
    .def_rw("stream", &LaunchParams::stream);

  module.def("make_kernel_config", &make_kernel_config, "target_sm"_a, "I"_a, "J"_a, "S"_a, "D"_a, "c_m"_a, "dtype"_a);

  module.def("dynamic_smem_bytes", &dynamic_smem_bytes, "config"_a);
  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      pair_weighted_averaging::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::pair_weighted_averaging
