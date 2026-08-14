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

#include "cubins/embedded_cubins.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::adaln_layernorm_sigmoid
{
namespace
{

/* Every shipped variant, taken from the generated registry rather than a
 * hand-written table, so the launch geometry cannot drift from itself.
 */
std::vector<KernelSpec> all_kernel_specs()
{
  std::vector<KernelSpec> specs;
  specs.reserve(embedded::kCubinCount);
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    embedded::CubinImage const& image = embedded::kCubins[index];
    specs.push_back(
      KernelSpec{
        image.cubin.target_sm,
        image.feature_dim,
        image.threads_per_row,
        image.num_threads,
        image.cluster_n,
      });
  }
  return specs;
}

std::vector<std::int32_t> all_feature_dims()
{
  std::vector<std::int32_t> dims;
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    std::int32_t const value = embedded::kCubins[index].feature_dim;
    bool seen = false;
    for (std::int32_t existing : dims)
      seen = seen || existing == value;
    if (!seen)
      dims.push_back(value);
  }
  return dims;
}

} // namespace

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "adaln_layernorm_sigmoid", "Direct CUDA Driver launcher for precompiled CuTeDSL AdaLN CUBINs");

  nb::enum_<DType>(module, "DType")
    .value("FLOAT16", DType::kFloat16)
    .value("BFLOAT16", DType::kBFloat16)
    .value("FLOAT32", DType::kFloat32);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("feature_dim", &KernelSpec::feature_dim)
    .def_ro("threads_per_row", &KernelSpec::threads_per_row)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_ro("cluster_n", &KernelSpec::cluster_n)
    .def_prop_ro("rows_per_block", [](KernelSpec const& spec) { return rows_per_block(spec); })
    .def_prop_ro("supports_direct_launch", [](KernelSpec const&) { return true; });

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("x", &LaunchParams::x)
    .def_rw("s_scale", &LaunchParams::s_scale)
    .def_rw("s_bias", &LaunchParams::s_bias)
    .def_rw("output", &LaunchParams::output)
    .def_rw("eps", &LaunchParams::eps)
    .def_rw("mult", &LaunchParams::mult)
    .def_rw("inner", &LaunchParams::inner)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &all_kernel_specs, "Return every shipped variant.");
  module.def("feature_dims", &all_feature_dims, "Return every compiled feature dimension (N).");

  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "dtype"_a,
    "feature_dim"_a,
    "threads_per_row"_a,
    "num_threads"_a);

  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      adaln_layernorm_sigmoid::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace bioir::cutedsl::adaln_layernorm_sigmoid
