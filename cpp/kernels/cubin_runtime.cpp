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

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace trtbnm::cutedsl
{
namespace
{

[[noreturn]] void throw_cuda_error(CUresult result, std::string const& operation)
{
  char const* name = nullptr;
  char const* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  throw std::runtime_error(
    operation + " failed: " + (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) + ": "
    + (text == nullptr ? "unknown CUDA Driver error" : text));
}

void check_cuda(CUresult result, std::string const& operation)
{
  if (result != CUDA_SUCCESS)
    throw_cuda_error(result, operation);
}

struct KernelCacheKey
{
  std::uintptr_t context;
  std::uintptr_t cubin_data;
  std::string kernel_symbol;

  bool operator<(KernelCacheKey const& other) const
  {
    return std::tie(context, cubin_data, kernel_symbol)
      < std::tie(other.context, other.cubin_data, other.kernel_symbol);
  }
};

struct CachedKernel
{
  cubin_kernel_t kernel;
  bool dynamic_smem_configured;
  bool non_portable_cluster_size_configured;
};

void configure_function_attributes(CachedKernel& cached, EmbeddedCubinImage const& image)
{
  if (!cached.dynamic_smem_configured)
  {
    CUresult const result = cuFuncSetAttribute(
      cached.kernel.function,
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      static_cast<int>(image.dynamic_smem_bytes));
    if (result != CUDA_SUCCESS)
    {
      throw_cuda_error(
        result, "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, " + std::string(image.variant_id) + ")");
    }
    cached.dynamic_smem_configured = true;
  }

  if (!cached.non_portable_cluster_size_configured)
  {
    CUresult const result
      = cuFuncSetAttribute(cached.kernel.function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
    if (result != CUDA_SUCCESS)
    {
      throw_cuda_error(
        result, "cuFuncSetAttribute(NON_PORTABLE_CLUSTER_SIZE_ALLOWED, " + std::string(image.variant_id) + ")");
    }
    cached.non_portable_cluster_size_configured = true;
  }
}

struct CubinPreloader
{
  std::string family;
  CubinPreloadFunction function;
};

std::mutex& preload_registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::vector<CubinPreloader>& preload_registry()
{
  static std::vector<CubinPreloader> registry;
  return registry;
}

std::size_t preload_registered_kernels(CUcontext context, std::int32_t device_sm)
{
  std::vector<CubinPreloader> preloaders;
  {
    std::lock_guard<std::mutex> lock(preload_registry_mutex());
    preloaders = preload_registry();
  }

  std::size_t loaded = 0;
  for (CubinPreloader const& preloader : preloaders)
    loaded += preloader.function(context, device_sm);
  return loaded;
}

CUcontext query_current_cuda_context()
{
  check_cuda(cuInit(0), "cuInit");
  CUcontext context = nullptr;
  check_cuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  return context;
}

std::int32_t query_cuda_sm(CUcontext context)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");

  CUdevice device{};
  check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
  int major = 0;
  int minor = 0;
  check_cuda(
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
    "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)");
  check_cuda(
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
    "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)");
  return major * 10 + minor;
}

std::int32_t query_cuda_device(CUcontext context)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");

  CUdevice device{};
  check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
  return static_cast<std::int32_t>(device);
}

std::int32_t query_cuda_multiprocessor_count(CUcontext context)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");

  CUdevice device{};
  check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
  int multiprocessor_count = 0;
  check_cuda(
    cuDeviceGetAttribute(&multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
    "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)");
  return multiprocessor_count;
}

} // namespace

bool cubin_supports_sm(EmbeddedCubinImage const& image, std::int32_t device_sm)
{
  if (image.supported_sms == nullptr || image.supported_sm_count == 0)
    return false;
  return std::find(image.supported_sms, image.supported_sms + image.supported_sm_count, device_sm)
    != image.supported_sms + image.supported_sm_count;
}

void check_cuda_driver(CUresult result, char const* operation)
{
  check_cuda(result, operation == nullptr ? "CUDA Driver operation" : operation);
}

CUcontext current_cuda_context()
{
  CUcontext const context = query_current_cuda_context();
  if (context == nullptr)
  {
    throw std::runtime_error("No current CUDA context. Create the framework CUDA context before loading CUBINs.");
  }
  return context;
}

std::int32_t cuda_sm_for_context(CUcontext context)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");

  static thread_local CUcontext cached_context = nullptr;
  static thread_local std::int32_t cached_sm = -1;
  if (cached_context != context)
  {
    cached_sm = query_cuda_sm(context);
    cached_context = context;
  }
  return cached_sm;
}

std::int32_t cuda_device_for_context(CUcontext context)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");

  static thread_local CUcontext cached_context = nullptr;
  static thread_local std::int32_t cached_device = -1;
  if (cached_context != context)
  {
    cached_device = query_cuda_device(context);
    cached_context = context;
  }
  return cached_device;
}

std::int32_t cuda_multiprocessor_count_for_context(CUcontext context)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");

  static thread_local CUcontext cached_context = nullptr;
  static thread_local std::int32_t cached_multiprocessor_count = -1;
  if (cached_context != context)
  {
    cached_multiprocessor_count = query_cuda_multiprocessor_count(context);
    cached_context = context;
  }
  return cached_multiprocessor_count;
}

std::int32_t current_cuda_sm()
{
  CUcontext const context = current_cuda_context();
  return cuda_sm_for_context(context);
}

cubin_kernel_t
load_embedded_kernel(CUcontext context, EmbeddedCubinImage const& image, bool configure_function_attributes_on_load)
{
  if (context == nullptr)
    throw std::invalid_argument("CUDA context must not be null");
  if (image.data == nullptr || image.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (image.launch_abi == nullptr || image.launch_abi[0] == '\0')
    throw std::invalid_argument("embedded CUBIN launch ABI must not be empty");
  if (image.variant_id == nullptr || image.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant ID must not be empty");
  if (image.kernel_symbol == nullptr || image.kernel_symbol[0] == '\0')
    throw std::invalid_argument("embedded CUBIN kernel symbol must not be empty");

  (void) cuda_sm_for_context(context);

  static std::mutex cache_mutex;
  static std::map<KernelCacheKey, CachedKernel> cache;

  KernelCacheKey const key{
    reinterpret_cast<std::uintptr_t>(context),
    reinterpret_cast<std::uintptr_t>(image.data),
    image.kernel_symbol,
  };
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto const existing = cache.find(key);
  if (existing != cache.end())
  {
    if (configure_function_attributes_on_load)
      configure_function_attributes(existing->second, image);
    return existing->second.kernel;
  }

  cubin_kernel_t loaded{};
  check_cuda(cuModuleLoadData(&loaded.module, image.data), "cuModuleLoadData(" + std::string(image.variant_id) + ")");
  CUresult const get_function_result = cuModuleGetFunction(&loaded.function, loaded.module, image.kernel_symbol);
  if (get_function_result != CUDA_SUCCESS)
  {
    cuModuleUnload(loaded.module);
    throw_cuda_error(get_function_result, "cuModuleGetFunction(" + std::string(image.variant_id) + ")");
  }
  CachedKernel cached{
    loaded,
    image.dynamic_smem_bytes <= 48U * 1024U,
    !image.non_portable_cluster_size_allowed,
  };
  if (configure_function_attributes_on_load)
  {
    try
    {
      configure_function_attributes(cached, image);
    }
    catch (...)
    {
      cuModuleUnload(loaded.module);
      throw;
    }
  }
  cache.emplace(key, cached);
  return loaded;
}

CubinPreloadRegistration::CubinPreloadRegistration(char const* family, CubinPreloadFunction function)
{
  if (family == nullptr || family[0] == '\0' || function == nullptr)
    throw std::invalid_argument("CUBIN preloader registration is incomplete");

  std::lock_guard<std::mutex> lock(preload_registry_mutex());
  auto const duplicate = std::find_if(
    preload_registry().begin(),
    preload_registry().end(),
    [family](CubinPreloader const& preloader) { return preloader.family == family; });
  if (duplicate != preload_registry().end())
  {
    throw std::logic_error("CUBIN preloader already registered for " + std::string(family));
  }
  preload_registry().push_back(CubinPreloader{family, function});
}

std::size_t preload_registered_kernels()
{
  CUcontext const context = current_cuda_context();
  return preload_registered_kernels(context, cuda_sm_for_context(context));
}

std::size_t preload_registered_kernels_if_context_active()
{
  if (cuInit(0) != CUDA_SUCCESS)
    return 0;
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS)
    return 0;
  if (context == nullptr)
    return 0;
  return preload_registered_kernels(context, cuda_sm_for_context(context));
}

} // namespace trtbnm::cutedsl
