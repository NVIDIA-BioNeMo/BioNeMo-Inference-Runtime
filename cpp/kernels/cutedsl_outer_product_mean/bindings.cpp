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

#include "outer_product_mean_registry.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::outer_product_mean
{
namespace
{

/* Every shipped config, taken from the generated registry rather than a
 * hand-written table: B/S/I/J are symbolic in this kernel, so the registry is
 * the only description of the launch geometry and cannot drift from itself.
 */
std::vector<KernelSpec> all_kernel_specs()
{
  return map_registry(
    embedded::registry(),
    [](embedded::CubinImage const& image)
    {
      return KernelSpec{
        image.cubin.target_sm,
        image.tile_i,
        image.tile_j,
        image.raster_factor,
        image.num_threads,
      };
    });
}

std::vector<std::string> all_config_identities()
{
  return map_registry(
    embedded::registry(),
    [](embedded::CubinImage const& image)
    { return std::string{image.config_identity == nullptr ? "" : image.config_identity}; });
}

} // namespace

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "outer_product_mean", "Direct CUDA Driver launcher for precompiled CuTeDSL outer-product-mean CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  module.attr("CHANNELS_C") = kChannelsC;
  module.attr("CHANNELS_D") = kChannelsD;
  module.attr("CHANNELS_CZ") = kChannelsCz;

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("tile_i", &KernelSpec::tile_i)
    .def_ro("tile_j", &KernelSpec::tile_j)
    .def_ro("raster_factor", &KernelSpec::raster_factor)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_prop_ro("supports_direct_launch", [](KernelSpec const&) { return true; });

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("has_bias", &KernelConfig::has_bias)
    .def_ro("norm_before", &KernelConfig::norm_before)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("a", &LaunchParams::a)
    .def_rw("b", &LaunchParams::b)
    .def_rw("num_mask", &LaunchParams::num_mask)
    .def_rw("weight", &LaunchParams::weight)
    .def_rw("bias", &LaunchParams::bias)
    .def_rw("output", &LaunchParams::output)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &all_kernel_specs, "Return every shipped tile configuration.");
  module.def("config_identities", &all_config_identities, "Return every shipped config identity string.");

  module.def(
    "make_kernel_config",
    [](std::int32_t target_sm, DType dtype, bool has_bias, bool norm_before, std::string const& config_identity)
    { return make_kernel_config(target_sm, dtype, has_bias, norm_before, config_identity.c_str()); },
    "target_sm"_a,
    "dtype"_a,
    "has_bias"_a,
    "norm_before"_a,
    "config_identity"_a);

  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      outer_product_mean::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::outer_product_mean
