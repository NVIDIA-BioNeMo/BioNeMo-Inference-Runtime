# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared torch helpers: RNG, tensors, auto-chunking, and kernel backends."""

from .auto_chunk import (
    AUTOCHUNK_MIN_AUTO,
    CHUNK_REGISTRY,
    CONTACT_PROB,
    DEFAULT_AUTOCHUNK_MIN,
    DEFAULT_AUTOCHUNK_MIN_REF,
    DEFAULT_MSA_AUTOCHUNK_MIN,
    DEFAULT_MSA_CHUNK_ROWS,
    DEFAULT_PAIR_CHUNK_ROWS,
    DEFAULT_PAIR_TRANSITION_POLICY,
    DIFFUSION_PAIR_TRANSITION,
    MSA_TRANSITION,
    OUTER_PRODUCT_MEAN,
    PAIR_TRANSITION,
    PAIR_WEIGHTED_AVERAGING,
    TRIANGLE_ATTENTION,
    ChunkPolicy,
    ChunkRegistry,
    chunk_apply,
    default_autochunk_min,
    iter_chunks,
)
from .common import (
    _deterministic_algorithms,
    commit_graph_safe_generator,
    make_graph_safe_generator,
    recursive_calling_load_weights,
    safe_generator,
)
from .kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelLibraryUnavailable,
    CuTeDSLKernelVariantUnavailable,
    KernelConfigBundle,
    KernelSourceUnavailable,
    get_config_file_name,
    launch_compiled_kernel,
    load_kernel_configs,
    load_source_module,
    populate_compiled_cache_from_library,
    require_kernel_backend,
    resolve_implementation,
    tensor_s1_d0,
    tensor_s2_d1,
    tensor_s3_d2,
    tensor_s4_d3,
)
from .tensor import (
    batched_gather,
    dict_map,
    dict_multimap,
    dist_one_hot,
    flatten_final_dims,
    masked_mean,
    pad_dim,
    permute_final_dims,
    tensor_tree_map,
    tree_map,
)

__all__ = [
    "AUTOCHUNK_MIN_AUTO",
    "CHUNK_REGISTRY",
    "CONTACT_PROB",
    "CuTeDSLKernelLibraryError",
    "CuTeDSLKernelLibraryExecutable",
    "CuTeDSLKernelLibraryUnavailable",
    "CuTeDSLKernelVariantUnavailable",
    "DEFAULT_AUTOCHUNK_MIN",
    "DEFAULT_AUTOCHUNK_MIN_REF",
    "DEFAULT_MSA_AUTOCHUNK_MIN",
    "DEFAULT_MSA_CHUNK_ROWS",
    "DEFAULT_PAIR_CHUNK_ROWS",
    "DEFAULT_PAIR_TRANSITION_POLICY",
    "DIFFUSION_PAIR_TRANSITION",
    "KernelConfigBundle",
    "KernelSourceUnavailable",
    "MSA_TRANSITION",
    "OUTER_PRODUCT_MEAN",
    "PAIR_TRANSITION",
    "PAIR_WEIGHTED_AVERAGING",
    "TRIANGLE_ATTENTION",
    "ChunkPolicy",
    "ChunkRegistry",
    "_deterministic_algorithms",
    "batched_gather",
    "chunk_apply",
    "commit_graph_safe_generator",
    "default_autochunk_min",
    "dict_map",
    "dict_multimap",
    "dist_one_hot",
    "flatten_final_dims",
    "get_config_file_name",
    "iter_chunks",
    "launch_compiled_kernel",
    "load_kernel_configs",
    "load_source_module",
    "make_graph_safe_generator",
    "masked_mean",
    "pad_dim",
    "permute_final_dims",
    "populate_compiled_cache_from_library",
    "recursive_calling_load_weights",
    "require_kernel_backend",
    "resolve_implementation",
    "safe_generator",
    "tensor_s1_d0",
    "tensor_s2_d1",
    "tensor_s3_d2",
    "tensor_s4_d3",
    "tensor_tree_map",
    "tree_map",
]
