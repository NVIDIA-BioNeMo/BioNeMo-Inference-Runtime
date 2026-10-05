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

namespace bioir::cutedsl::trimul_kf_k3
{

void bind(nb::module_& parent)
{
  nb::module_ module
    = parent.def_submodule("trimul_kf_k3", "Direct CUDA Driver launcher for precompiled SM90 TriMul KF K3 CUBINs");

  nb::enum_<DType>(module, "DType").value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("kernel_sm", &KernelSpec::kernel_sm)
    .def_ro("C", &KernelSpec::C)
    .def_ro("D", &KernelSpec::D)
    .def_ro("kernel_variant", &KernelSpec::kernel_variant)
    .def_ro("residual", &KernelSpec::residual)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_ro("tile_m", &KernelSpec::tile_m)
    .def_ro("tile_ctas", &KernelSpec::tile_ctas);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("prod", &LaunchParams::prod)
    .def_rw("x", &LaunchParams::x)
    .def_rw("w_out", &LaunchParams::w_out)
    .def_rw("w_gate_out", &LaunchParams::w_gate_out)
    .def_rw("vec_out", &LaunchParams::vec_out)
    .def_rw("stats", &LaunchParams::stats)
    .def_rw("seqlen", &LaunchParams::seqlen)
    .def_rw("output", &LaunchParams::output)
    .def_rw("rows", &LaunchParams::rows)
    .def_rw("n", &LaunchParams::n)
    .def_rw("nb", &LaunchParams::nb)
    .def_rw("eps", &LaunchParams::eps)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &kernel_specs, "Return every registered trimul KF K3 variant.");
  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "dtype"_a,
    "C"_a,
    "D"_a,
    "kernel_variant"_a,
    "residual"_a);
  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      trimul_kf_k3::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, and stream.");
}

} // namespace bioir::cutedsl::trimul_kf_k3
