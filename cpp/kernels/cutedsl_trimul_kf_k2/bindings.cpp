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

namespace bioir::cutedsl::trimul_kf_k2
{

void bind(nb::module_& parent)
{
  nb::module_ module
    = parent.def_submodule("trimul_kf_k2", "Direct CUDA Driver launcher for precompiled SM90 TriMul KF K2 CUBINs");

  nb::enum_<DType>(module, "DType").value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("kernel_sm", &KernelSpec::kernel_sm)
    .def_ro("outgoing", &KernelSpec::outgoing)
    .def_ro("kernel_variant", &KernelSpec::kernel_variant)
    .def_ro("tile_m", &KernelSpec::tile_m)
    .def_ro("tile_n", &KernelSpec::tile_n)
    .def_ro("cluster_m", &KernelSpec::cluster_m)
    .def_ro("defer_kmin", &KernelSpec::defer_kmin)
    .def_ro("split_epi", &KernelSpec::split_epi)
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
    .def_rw("a", &LaunchParams::a)
    .def_rw("b", &LaunchParams::b)
    .def_rw("prod", &LaunchParams::prod)
    .def_rw("n", &LaunchParams::n)
    .def_rw("l", &LaunchParams::l)
    .def_rw("ab_pitch", &LaunchParams::ab_pitch)
    .def_rw("ab_plane", &LaunchParams::ab_plane)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &kernel_specs, "Return every registered trimul KF K2 variant.");
  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "dtype"_a,
    "outgoing"_a,
    "kernel_variant"_a,
    "tile_n"_a,
    "cluster_m"_a,
    "defer_kmin"_a,
    "split_epi"_a);
  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      trimul_kf_k2::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, and stream.");
}

} // namespace bioir::cutedsl::trimul_kf_k2
