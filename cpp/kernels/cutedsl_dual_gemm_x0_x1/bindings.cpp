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

namespace nb = nanobind;
using namespace nb::literals;

namespace trtbnm::cutedsl::dual_gemm_x0_x1
{

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "dual_gemm_x0_x1", "Direct CUDA Driver launcher for precompiled CuTeDSL dual-GEMM x0_x1 CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("kernel_sm", &KernelSpec::kernel_sm)
    .def_ro("K", &KernelSpec::K)
    .def_ro("N", &KernelSpec::N)
    .def_ro("bucket", &KernelSpec::bucket)
    .def_ro("has_bias", &KernelSpec::has_bias)
    .def_ro("tile_m", &KernelSpec::tile_m)
    .def_ro("tile_n", &KernelSpec::tile_n)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_ro("raster_factor", &KernelSpec::raster_factor)
    .def_prop_ro("supports_direct_launch", [](KernelSpec const&) { return true; });

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("has_bias", &KernelConfig::has_bias)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("x0", &LaunchParams::x0)
    .def_rw("x1", &LaunchParams::x1)
    .def_rw("w0", &LaunchParams::w0)
    .def_rw("w1", &LaunchParams::w1)
    .def_rw("bias0", &LaunchParams::bias0)
    .def_rw("bias1", &LaunchParams::bias1)
    .def_rw("out", &LaunchParams::out)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &kernel_specs, "Return every registered SM/shape/bucket/bias launch configuration.");

  module.def("make_kernel_config", &make_kernel_config, "target_sm"_a, "K"_a, "N"_a, "S"_a, "dtype"_a, "has_bias"_a);

  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      dual_gemm_x0_x1::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace trtbnm::cutedsl::dual_gemm_x0_x1
