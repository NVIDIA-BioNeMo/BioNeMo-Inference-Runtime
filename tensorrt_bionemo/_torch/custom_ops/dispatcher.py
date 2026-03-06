# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch

from tensorrt_bionemo._torch.custom_ops.base import IGNORE_BENCHMARK_SCORE

from .cuequiv_dual_gemm import CUEQUIV_DUAL_GEMM_KERNELS
from .cutlass_dual_gemm import CUTLASS_DUAL_GEMM_KERNELS

_BACKEND_REGISTRIES = [CUTLASS_DUAL_GEMM_KERNELS, CUEQUIV_DUAL_GEMM_KERNELS]

_OPS_IMPLS_MAP: dict[str, list] = {}
for _registry in _BACKEND_REGISTRIES:
    for _op_name, _impl in _registry.items():
        _OPS_IMPLS_MAP.setdefault(_op_name, []).append(_impl)

_IMPL_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX_SIZE = 1024


def _tensor_key(t: Optional[torch.Tensor]) -> tuple:
    if t is None:
        return (None, )
    return (t.shape, t.dtype, t.device)


def _make_cache_key(ops_name: str, *args, **kwargs) -> tuple:
    key = [ops_name]
    for a in args:
        key.append(_tensor_key(a) if isinstance(a, torch.Tensor) else a)
    for k in sorted(kwargs):
        v = kwargs[k]
        key.append((k, _tensor_key(v) if isinstance(v, torch.Tensor) else v))
    return tuple(key)


def _find_best_impl(ops_name: str, *args, **kwargs):
    best_impl = None
    best_score = IGNORE_BENCHMARK_SCORE
    for impl in _OPS_IMPLS_MAP.get(ops_name, []):
        if not impl.is_supported(*args, **kwargs):
            continue
        score = impl.get_benchmark_score(*args, **kwargs)
        if score == IGNORE_BENCHMARK_SCORE:
            continue
        if score > best_score:
            best_score = score
            best_impl = impl
    return best_impl


def get_custom_ops_impl(ops_name: str, *args, **kwargs) -> Optional[Callable]:
    """Return a callable for the best-performing supported implementation.

    Scans all registered backends for *ops_name*, keeps those whose
    ``is_supported`` returns ``True``, ranks them by ``get_benchmark_score``,
    and returns the one with the highest score.  Implementations that report
    ``IGNORE_BENCHMARK_SCORE`` (-1) are skipped.

    Results are cached by tensor metadata (shape / dtype / device) so the
    selection logic only runs once per unique configuration.  The cache holds
    up to ``_CACHE_MAX_SIZE`` (default 1024) entries with LRU eviction.
    """
    cache_key = _make_cache_key(ops_name, *args, **kwargs)
    if cache_key in _IMPL_CACHE:
        _IMPL_CACHE.move_to_end(cache_key)
        best_impl = _IMPL_CACHE[cache_key]
    else:
        best_impl = _find_best_impl(ops_name, *args, **kwargs)
        _IMPL_CACHE[cache_key] = best_impl
        if len(_IMPL_CACHE) > _CACHE_MAX_SIZE:
            _IMPL_CACHE.popitem(last=False)
    if best_impl is None:
        return None
    return partial(best_impl.apply, *args, **kwargs)
