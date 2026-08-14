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

namespace bioir::cutedsl::dual_gemm_x_x
{

void bind(nb::module_& parent)
{
  nb::module_ module
    = parent.def_submodule("dual_gemm_x_x", "Direct CUDA Driver launcher for precompiled CuTeDSL dual-GEMM x_x CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("target_sm", &KernelConfig::target_sm)
    .def_ro("K", &KernelConfig::K)
    .def_ro("N", &KernelConfig::N)
    .def_ro("bucket", &KernelConfig::bucket)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("transpose_out", &KernelConfig::transpose_out)
    .def_ro("has_bias", &KernelConfig::has_bias)
    .def_ro("has_mask", &KernelConfig::has_mask)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("x", &LaunchParams::x)
    .def_rw("w0", &LaunchParams::w0)
    .def_rw("w1", &LaunchParams::w1)
    .def_rw("output", &LaunchParams::output)
    .def_rw("bias0", &LaunchParams::bias0)
    .def_rw("bias1", &LaunchParams::bias1)
    .def_rw("actual_seqlen", &LaunchParams::actual_seqlen)
    .def_rw("i_dim", &LaunchParams::i_dim)
    .def_rw("stream", &LaunchParams::stream);

  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "K"_a,
    "N"_a,
    "S"_a,
    "dtype"_a,
    "transpose_out"_a,
    "has_bias"_a,
    "has_mask"_a);

  module.def("dynamic_smem_bytes", &dynamic_smem_bytes, "config"_a);
  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      dual_gemm_x_x::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::dual_gemm_x_x
