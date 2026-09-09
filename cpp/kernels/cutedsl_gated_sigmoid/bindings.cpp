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

#include "gated_sigmoid_registry.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::gated_sigmoid
{
namespace
{

/* Every shipped tile, taken from the generated registry rather than a
 * hand-written table: K and N are symbolic in this kernel, so the registry is
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
        image.m_block_size,
        image.n_block_size,
        image.k_block_size,
        image.num_stages,
        image.raster_factor,
        {image.atom_layout_mnk[0], image.atom_layout_mnk[1], image.atom_layout_mnk[2]},
        image.num_threads,
      };
    });
}

} // namespace

void bind(nb::module_& parent)
{
  nb::module_ module
    = parent.def_submodule("gated_sigmoid", "Direct CUDA Driver launcher for precompiled CuTeDSL gated-sigmoid CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("m_block_size", &KernelSpec::m_block_size)
    .def_ro("n_block_size", &KernelSpec::n_block_size)
    .def_ro("k_block_size", &KernelSpec::k_block_size)
    .def_ro("num_stages", &KernelSpec::num_stages)
    .def_ro("raster_factor", &KernelSpec::raster_factor)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_prop_ro(
      "atom_layout_mnk",
      [](KernelSpec const& spec)
      {
        return std::array<std::int32_t, 3>{spec.atom_layout_mnk[0], spec.atom_layout_mnk[1], spec.atom_layout_mnk[2]};
      })
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
    .def_rw("s", &LaunchParams::s)
    .def_rw("weight", &LaunchParams::weight)
    .def_rw("bias", &LaunchParams::bias)
    .def_rw("mha_out", &LaunchParams::mha_out)
    .def_rw("output", &LaunchParams::output)
    .def_rw("mult", &LaunchParams::mult)
    .def_rw("inner", &LaunchParams::inner)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &all_kernel_specs, "Return every shipped tile configuration.");

  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "dtype"_a,
    "has_bias"_a,
    "m_block_size"_a,
    "n_block_size"_a,
    "k_block_size"_a,
    "num_stages"_a,
    "raster_factor"_a,
    "atom_layout_m"_a,
    "atom_layout_n"_a,
    "atom_layout_k"_a);

  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      gated_sigmoid::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::gated_sigmoid
