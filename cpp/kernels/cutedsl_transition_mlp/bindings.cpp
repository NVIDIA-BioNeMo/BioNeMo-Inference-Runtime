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
#include <nanobind/stl/vector.h>

#include <cstdint>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::transition_mlp
{

void bind(nb::module_& parent)
{
  nb::module_ module
    = parent.def_submodule("transition_mlp", "Direct CUDA Driver launcher for precompiled transition MLP CUBINs");

  nb::enum_<DType>(module, "DType").value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("kernel_sm", &KernelSpec::kernel_sm)
    .def_ro("width", &KernelSpec::width)
    .def_ro("hidden", &KernelSpec::hidden)
    .def_ro("bucket", &KernelSpec::bucket)
    .def_ro("is_silu_gate", &KernelSpec::is_silu_gate)
    .def_ro("has_bias", &KernelSpec::has_bias)
    .def_ro("has_mask", &KernelSpec::has_mask)
    .def_ro("has_residual", &KernelSpec::has_residual)
    .def_ro("tile_m", &KernelSpec::tile_m)
    .def_ro("num_threads", &KernelSpec::num_threads);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("x", &LaunchParams::x)
    .def_rw("w1", &LaunchParams::w1)
    .def_rw("b1", &LaunchParams::b1)
    .def_rw("w2", &LaunchParams::w2)
    .def_rw("b2", &LaunchParams::b2)
    .def_rw("residual", &LaunchParams::residual)
    .def_rw("mask", &LaunchParams::mask)
    .def_rw("output", &LaunchParams::output)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &kernel_specs, "Return every registered transition MLP variant.");
  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "dtype"_a,
    "is_silu_gate"_a,
    "has_bias"_a,
    "has_mask"_a,
    "has_residual"_a,
    "width"_a,
    "hidden"_a,
    "bucket"_a);
  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      transition_mlp::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::transition_mlp
