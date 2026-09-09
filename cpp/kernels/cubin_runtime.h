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

/* Shared embedded-CUBIN loading, caching, and preloading runtime.
 */

#ifndef BIOIR_CPP_KERNELS_CUBIN_RUNTIME_H_
#define BIOIR_CPP_KERNELS_CUBIN_RUNTIME_H_

#include "cubin_launch.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioir::cutedsl
{

struct EmbeddedCubinImage
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::uint32_t dynamic_smem_bytes;
  bool non_portable_cluster_size_allowed;
  char const* launch_abi;
  std::int32_t const* supported_sms;
  std::size_t supported_sm_count;
  char const* variant_id;
  char const* kernel_symbol;
  void const* data;
  std::size_t size;
};

bool cubin_supports_sm(EmbeddedCubinImage const& image, std::int32_t device_sm);

inline constexpr std::size_t kTmaMaxRank = 5;

/* Host-side TMA descriptor recipe baked into a CUBIN at build time.
 *
 * Arch-generic: any SM90+ native kernel encodes its descriptors from these
 * fields, so families share this definition and only the set of descriptors
 * they carry (named per tensor operand) stays family-specific.
 */
struct TmaDescriptorInfo
{
  CUtensorMapDataType data_type;
  std::uint32_t rank;
  std::uint32_t global_dim_order[kTmaMaxRank];
  std::uint32_t box_dims[kTmaMaxRank];
  std::uint32_t element_strides[kTmaMaxRank];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2_promotion;
  CUtensorMapFloatOOBfill oob_fill;
};

void check_cuda_driver(CUresult result, char const* operation);

CUcontext current_cuda_context();

std::int32_t cuda_sm_for_context(CUcontext context);

std::int32_t cuda_device_for_context(CUcontext context);

std::int32_t cuda_multiprocessor_count_for_context(CUcontext context);

std::int32_t current_cuda_sm();

cubin_kernel_t
load_embedded_kernel(CUcontext context, EmbeddedCubinImage const& image, bool configure_function_attributes = true);

/* Load every generated-registry image this device can run.
 *
 * Each family's generated `embedded::registry()` returns its own
 * `{images, count}` view over its own `CubinImage` type, but the preload walk
 * over that view is identical, so it lives here once. Families that need more
 * than SM compatibility (pair-weighted averaging filters on its runtime axes
 * too) pass their own predicate.
 */
template <typename RegistryView, typename CompatiblePredicate>
std::size_t preload_registry_kernels(
  CUcontext context, std::int32_t device_sm, RegistryView const& registry, CompatiblePredicate compatible)
{
  std::size_t loaded = 0;
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    auto const& image = registry.images[index];
    if (!compatible(image, device_sm))
      continue;
    (void) load_embedded_kernel(context, image.cubin, false);
    ++loaded;
  }
  return loaded;
}

template <typename RegistryView>
std::size_t preload_registry_kernels(CUcontext context, std::int32_t device_sm, RegistryView const& registry)
{
  return preload_registry_kernels(
    context,
    device_sm,
    registry,
    [](auto const& image, std::int32_t sm) { return cubin_supports_sm(image.cubin, sm); });
}

/* Project every generated-registry image through `transform`, in registry order.
 *
 * The nanobind bindings and the launchers publish several per-family views of
 * the registry (kernel specs, config identities); only the projection differs.
 */
template <typename RegistryView, typename Transform>
auto map_registry(RegistryView const& registry, Transform transform)
{
  std::vector<decltype(transform(registry.images[0]))> mapped;
  mapped.reserve(registry.count);
  for (std::size_t index = 0; index < registry.count; ++index)
    mapped.push_back(transform(registry.images[index]));
  return mapped;
}

using CubinPreloadFunction = std::size_t (*)(CUcontext context, std::int32_t device_sm);

class CubinPreloadRegistration
{
  public:
  CubinPreloadRegistration(char const* family, CubinPreloadFunction function);
};

std::size_t preload_registered_kernels();

std::size_t preload_registered_kernels_if_context_active();

} // namespace bioir::cutedsl

#endif /* BIOIR_CPP_KERNELS_CUBIN_RUNTIME_H_ */
