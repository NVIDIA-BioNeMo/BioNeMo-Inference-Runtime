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

#include "window_attention_registry.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::window_attention
{

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "window_attention", "Direct CUDA Driver launcher for precompiled CuTeDSL windowed-attention CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("target_sm", &KernelConfig::target_sm)
    .def_ro("head_dim", &KernelConfig::head_dim)
    .def_ro("n_queries", &KernelConfig::n_queries)
    .def_ro("n_keys", &KernelConfig::n_keys)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; })
    .def_prop_ro("kernel_sm", [](KernelConfig const& config) { return config.cubin.kernel_sm; })
    .def_prop_ro("launch_abi", [](KernelConfig const& config) { return config.cubin.launch_abi; })
    .def_prop_ro("num_threads", [](KernelConfig const& config) { return config.embedded_image->num_threads; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("q", &LaunchParams::q)
    .def_rw("k", &LaunchParams::k)
    .def_rw("v", &LaunchParams::v)
    .def_rw("bias", &LaunchParams::bias)
    .def_rw("output", &LaunchParams::output)
    .def_rw("softmax_scale_log2", &LaunchParams::softmax_scale_log2)
    .def_rw("softmax_scale", &LaunchParams::softmax_scale)
    .def_rw("stream", &LaunchParams::stream);

  module.def(
    "make_kernel_config", &make_kernel_config, "target_sm"_a, "head_dim"_a, "n_queries"_a, "n_keys"_a, "dtype"_a);

  module.def("dynamic_smem_bytes", &dynamic_smem_bytes, "config"_a);
  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      window_attention::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::window_attention
