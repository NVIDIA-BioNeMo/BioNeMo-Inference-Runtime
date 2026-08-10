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

#include "cubin_runtime.h"
#include "cutedsl_tensor_abi.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace trtbnm::cutedsl::triangle_attention
{

void bind(nb::module_& module);

} // namespace trtbnm::cutedsl::triangle_attention

namespace trtbnm::cutedsl::pairwise_attention
{

void bind(nb::module_& module);

} // namespace trtbnm::cutedsl::pairwise_attention

namespace trtbnm::cutedsl::dual_gemm_x_x
{

void bind(nb::module_& module);

} // namespace trtbnm::cutedsl::dual_gemm_x_x

namespace
{

template <std::size_t N>
void bind_tensor_view(nb::module_& module, char const* name)
{
  using View = trtbnm::cutedsl::TensorView<N>;
  nb::class_<View>(module, name)
    .def(
      nb::init<std::uint64_t, std::array<std::int32_t, N>, std::array<std::int64_t, N - 1>, std::int32_t>(),
      "data"_a,
      "shape"_a,
      "strides"_a,
      "device"_a = trtbnm::cutedsl::kUnknownDevice)
    .def_rw("data", &View::data)
    .def_rw("shape", &View::shape)
    .def_rw("strides", &View::strides)
    .def_rw("device", &View::device);
}

} // namespace

NB_MODULE(_cutedsl_kernels, module)
{
  using namespace trtbnm::cutedsl;

  module.doc() = "CUDA Driver runtime for precompiled TensorRT-BioNeMo CuTeDSL kernels";

  nb::class_<EmbeddedCubinImage>(module, "Cubin")
    .def_ro("target_sm", &EmbeddedCubinImage::target_sm)
    .def_ro("kernel_sm", &EmbeddedCubinImage::kernel_sm)
    .def_ro("dynamic_smem_bytes", &EmbeddedCubinImage::dynamic_smem_bytes)
    .def_ro("non_portable_cluster_size_allowed", &EmbeddedCubinImage::non_portable_cluster_size_allowed)
    .def_ro("launch_abi", &EmbeddedCubinImage::launch_abi)
    .def_prop_ro(
      "supported_sms",
      [](EmbeddedCubinImage const& image)
      {
        if (image.supported_sms == nullptr)
          return std::vector<std::int32_t>{};
        return std::vector<std::int32_t>(image.supported_sms, image.supported_sms + image.supported_sm_count);
      })
    .def_ro("variant_id", &EmbeddedCubinImage::variant_id)
    .def_ro("kernel_symbol", &EmbeddedCubinImage::kernel_symbol)
    .def_ro("size", &EmbeddedCubinImage::size);

  module.attr("UNKNOWN_DEVICE") = kUnknownDevice;

  bind_tensor_view<1>(module, "Tensor1View");
  bind_tensor_view<2>(module, "Tensor2View");
  bind_tensor_view<3>(module, "Tensor3View");
  bind_tensor_view<4>(module, "Tensor4View");

  module.def(
    "preload",
    []()
    {
      nb::gil_scoped_release release;
      return preload_registered_kernels();
    },
    "Load every registered CUBIN compatible with the current CUDA device.");

  triangle_attention::bind(module);
  pairwise_attention::bind(module);
  dual_gemm_x_x::bind(module);
  (void) preload_registered_kernels_if_context_active();
}
