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
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace trtbnm::cutedsl::pairwise_attention
{
namespace
{

std::vector<KernelSpec> all_kernel_specs()
{
  return {
    kKernelSpecs.begin(),
    kKernelSpecs.end(),
  };
}

} // namespace

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "pairwise_attention", "Direct CUDA Driver launcher for precompiled CuTeDSL pairwise-attention CUBINs");

  nb::enum_<DType>(module, "DType").value("FLOAT16", DType::kFloat16).value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelSpecSM80>(module, "KernelSpecSM80")
    .def_ro("target_sm", &KernelSpecSM80::target_sm)
    .def_ro("head_dim", &KernelSpecSM80::head_dim)
    .def_ro("bucket", &KernelSpecSM80::bucket)
    .def_ro("tile_m", &KernelSpecSM80::tile_m)
    .def_ro("tile_n", &KernelSpecSM80::tile_n)
    .def_ro("num_threads", &KernelSpecSM80::num_threads)
    .def_ro("swizzle_b", &KernelSpecSM80::swizzle_b)
    .def_ro("load_bias_before_gemm", &KernelSpecSM80::load_bias_before_gemm)
    .def_prop_ro("supports_direct_launch", [](KernelSpecSM80 const&) { return true; });

  nb::class_<KernelSpecSM90>(module, "KernelSpecSM90")
    .def_ro("target_sm", &KernelSpecSM90::target_sm)
    .def_ro("head_dim", &KernelSpecSM90::head_dim)
    .def_ro("bucket", &KernelSpecSM90::bucket)
    .def_ro("tile_m", &KernelSpecSM90::tile_m)
    .def_ro("tile_n", &KernelSpecSM90::tile_n)
    .def_ro("num_threads", &KernelSpecSM90::num_threads)
    .def_ro("mma_m", &KernelSpecSM90::mma_m)
    .def_ro("mma_n", &KernelSpecSM90::mma_n)
    .def_ro("kv_stage", &KernelSpecSM90::kv_stage)
    .def_ro("raster_factor", &KernelSpecSM90::raster_factor)
    .def_ro("persistent", &KernelSpecSM90::persistent)
    .def_prop_ro("supports_direct_launch", [](KernelSpecSM90 const&) { return true; });

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("packed_output", &KernelConfig::packed_output)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("cubin_size", [](KernelConfig const& config) { return config.cubin.size; })
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("q", &LaunchParams::q)
    .def_rw("k", &LaunchParams::k)
    .def_rw("v", &LaunchParams::v)
    .def_rw("actual_s_kv", &LaunchParams::actual_s_kv)
    .def_rw("bias", &LaunchParams::bias)
    .def_rw("output", &LaunchParams::output)
    .def_rw("lse", &LaunchParams::lse)
    .def_rw("softmax_scale", &LaunchParams::softmax_scale)
    .def_rw("softmax_scale_log2", &LaunchParams::softmax_scale_log2)
    .def_rw("mult", &LaunchParams::mult)
    .def_rw("stream", &LaunchParams::stream);

  module.def(
    "kernel_specs", &all_kernel_specs, "Return all registered SM/head-dimension/bucket launch configurations.");

  module.def(
    "make_kernel_config", &make_kernel_config, "target_sm"_a, "head_dim"_a, "S"_a, "dtype"_a, "packed_output"_a);

  module.def("current_cuda_sm", &current_cuda_sm);

  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      pairwise_attention::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, strides, and stream.");
}

} // namespace trtbnm::cutedsl::pairwise_attention
