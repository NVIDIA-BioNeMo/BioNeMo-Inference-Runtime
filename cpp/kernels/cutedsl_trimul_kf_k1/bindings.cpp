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
#include <new>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::cutedsl::trimul_kf_k1
{

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule(
    "trimul_kf_k1", "Direct CUDA Driver launcher for precompiled SM90 and SM100/SM103 TriMul KF K1 CUBINs");

  nb::enum_<DType>(module, "DType").value("BFLOAT16", DType::kBFloat16);

  nb::class_<KernelTile>(module, "KernelTile")
    .def(nb::init<>())
    .def(
      "__init__",
      [](
        KernelTile* tile,
        std::uint32_t bnc,
        std::uint32_t epi_n,
        std::uint32_t ctas,
        std::uint32_t w_slots,
        std::uint32_t s_stages,
        std::uint32_t store_warps,
        std::uint32_t epi_groups)
      { new (tile) KernelTile{bnc, epi_n, ctas, w_slots, s_stages, store_warps, epi_groups}; },
      "bnc"_a,
      "epi_n"_a,
      "ctas"_a,
      "w_slots"_a,
      "s_stages"_a,
      "store_warps"_a,
      "epi_groups"_a)
    .def_ro("bnc", &KernelTile::bnc)
    .def_ro("epi_n", &KernelTile::epi_n)
    .def_ro("ctas", &KernelTile::ctas)
    .def_ro("w_slots", &KernelTile::w_slots)
    .def_ro("s_stages", &KernelTile::s_stages)
    .def_ro("store_warps", &KernelTile::store_warps)
    .def_ro("epi_groups", &KernelTile::epi_groups);

  nb::class_<KernelSpec>(module, "KernelSpec")
    .def_ro("target_sm", &KernelSpec::target_sm)
    .def_ro("kernel_sm", &KernelSpec::kernel_sm)
    .def_ro("C", &KernelSpec::C)
    .def_ro("D", &KernelSpec::D)
    .def_ro("kernel_variant", &KernelSpec::kernel_variant)
    .def_ro("padded", &KernelSpec::padded)
    .def_ro("num_threads", &KernelSpec::num_threads)
    .def_ro("tile", &KernelSpec::tile);

  nb::class_<KernelConfig>(module, "KernelConfig")
    .def_ro("spec", &KernelConfig::spec)
    .def_ro("dtype", &KernelConfig::dtype)
    .def_ro("cubin", &KernelConfig::cubin)
    .def_prop_ro("dynamic_smem_bytes", &dynamic_smem_bytes)
    .def_prop_ro("variant_id", [](KernelConfig const& config) { return config.cubin.variant_id; })
    .def_prop_ro("kernel_symbol", [](KernelConfig const& config) { return config.cubin.kernel_symbol; });

  nb::class_<LaunchParams>(module, "LaunchParams")
    .def(nb::init<>())
    .def_rw("x", &LaunchParams::x)
    .def_rw("seqlen", &LaunchParams::seqlen)
    .def_rw("w_in", &LaunchParams::w_in)
    .def_rw("w_gate_in", &LaunchParams::w_gate_in)
    .def_rw("vec_in", &LaunchParams::vec_in)
    .def_rw("a", &LaunchParams::a)
    .def_rw("b", &LaunchParams::b)
    .def_rw("planes", &LaunchParams::planes)
    .def_rw("stats", &LaunchParams::stats)
    .def_rw("rows", &LaunchParams::rows)
    .def_rw("n", &LaunchParams::n)
    .def_rw("nb", &LaunchParams::nb)
    .def_rw("eps", &LaunchParams::eps)
    .def_rw("stream", &LaunchParams::stream);

  module.def("kernel_specs", &kernel_specs, "Return every registered trimul KF K1 variant.");
  module.def(
    "make_kernel_config",
    &make_kernel_config,
    "target_sm"_a,
    "dtype"_a,
    "C"_a,
    "D"_a,
    "kernel_variant"_a,
    "padded"_a = false,
    "tile"_a = KernelTile{});
  module.def(
    "launch",
    [](KernelConfig const& config, LaunchParams const& params)
    {
      nb::gil_scoped_release release;
      trimul_kf_k1::launch(config, params);
    },
    "config"_a,
    "params"_a,
    "Launch using prepacked raw pointers, shapes, and stream.");
}

} // namespace bioir::cutedsl::trimul_kf_k1
