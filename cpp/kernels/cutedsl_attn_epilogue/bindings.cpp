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

namespace bioir::cutedsl::attn_epilogue
{

void bind(nb::module_& parent)
{
  nb::module_ module
    = parent.def_submodule("attn_epilogue", "Direct CUDA Driver launcher for precompiled attention epilogue CUBINs");

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("kernel_sm", &KernelSpec::kernel_sm)
    .def_ro("heads", &KernelSpec::heads)
    .def_ro("head_dim", &KernelSpec::head_dim)
    .def_ro("channels", &KernelSpec::channels)
    .def_ro("has_bias", &KernelSpec::has_bias)
    .def_ro("has_output_gate", &KernelSpec::has_output_gate)
    .def_ro("has_residual", &KernelSpec::has_residual)
    .def_ro("tile_j", &KernelSpec::tile_j)
    .def_ro("tile_n", &KernelSpec::tile_n)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_ro("bucket", &KernelSpec::bucket)
    .def_ro("pairs", &KernelSpec::pairs);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", [](KernelConfig const& config) { return config.cubin.dynamic_smem_bytes; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("o", &LaunchParams::o)
    .def_rw("g", &LaunchParams::g)
    .def_rw("w", &LaunchParams::w)
    .def_rw("b", &LaunchParams::b)
    .def_rw("d", &LaunchParams::d)
    .def_rw("z", &LaunchParams::z)
    .def_rw("y", &LaunchParams::y)
    .def_rw("stream", &LaunchParams::stream);

  nb::class_<TokenLaunchParams>(module, "TokenLaunchParams")
    .def(nb::init<>())
    .def_rw("o", &TokenLaunchParams::o)
    .def_rw("g", &TokenLaunchParams::g)
    .def_rw("w", &TokenLaunchParams::w)
    .def_rw("b", &TokenLaunchParams::b)
    .def_rw("d", &TokenLaunchParams::d)
    .def_rw("z", &TokenLaunchParams::z)
    .def_rw("y", &TokenLaunchParams::y)
    .def_rw("columns", &TokenLaunchParams::columns)
    .def_rw("gate_repeat", &TokenLaunchParams::gate_repeat)
    .def_rw("max_clusters", &TokenLaunchParams::max_clusters)
    .def_rw("stream", &TokenLaunchParams::stream);

  module.def("kernel_specs", &kernel_specs, "Return every registered SM and layer-shape launch configuration.");
  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "heads"_a,
    "head_dim"_a,
    "channels"_a,
    "has_bias"_a = false,
    "has_output_gate"_a = false,
    "rows"_a = 0,
    "has_residual"_a = true,
    "pairs"_a = false);
  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      attn_epilogue::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
  module.def(
    "launch_token",
    [](KernelConfig const& config, TokenLaunchParams const& params)
    {
      nb::gil_scoped_release release;
      attn_epilogue::launch_token(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch a Blackwell token-kernel CUBIN on flat row views.");
  module.def(
    "max_active_clusters",
    &max_active_clusters,
    "config"_a,
    "Return the token-kernel CUBIN's clusters co-resident on the current device.");
}

} // namespace bioir::cutedsl::attn_epilogue
