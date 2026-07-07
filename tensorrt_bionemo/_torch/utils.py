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

from typing import Callable

import contextlib
import torch
from torch import nn


def recursive_calling_load_weights(module: nn.Module,
                                   weights: dict,
                                   filter_func: Callable = None) -> set[str]:
    """
    DFS calling load_weights for the module.
    Args:
        module: The module to load weights for.
        weights: The weights to load.
        filter_func: The function to filter the modules to load weights for.
    Returns:
        The set of loaded weights.
    """
    loaded_weight = set()

    for name, module in module.named_modules():
        if filter_func is not None and filter_func(name, module):
            continue
        if len(module._parameters) > 0:
            # Uncomment to debug
            # print(f"loading for: {name}")
            try:
                if hasattr(module, 'load_weights'):
                    module.load_weights(weights=weights[name])
                else:
                    # Uncomment to debug
                    # print(f" use copy_ to load {name}")
                    module_weights = weights[name][0]
                    for n, p in module._parameters.items():
                        if p is not None:
                            weight = module_weights[n][:]
                            if p.dtype != weight.dtype:
                                weight = weight.to(p.dtype)
                            p.data.copy_(weight)

            except Exception as e:
                print(name)
                raise e
        loaded_weight.add(name)
    return loaded_weight


@contextlib.contextmanager
def _deterministic_algorithms():
    """Force deterministic CUDA algorithms within the block, restoring the
    previous setting on exit.

    CUDA ``scatter_add_`` (like ``index_add_``) accumulates colliding
    destination indices with ``atomicAdd`` in an unspecified order. Because
    floating-point addition is not associative, the accumulated result is
    **not** reproducible run-to-run -- *regardless of dtype* (fp32 only shrinks
    the per-call noise to ~1e-6, it does not remove it). Inside the diffusion
    rollout this tiny per-step noise compounds over hundreds of denoising steps
    into visibly divergent structures. Forcing PyTorch's deterministic scatter
    kernel (a fixed reduction order) removes the run-to-run noise.

    Scoped to just the scatter so the (slower) deterministic kernel only
    affects this aggregation, not the rest of the forward; the global flag is
    saved and restored so callers see no change in PyTorch state.
    """
    prev_enabled = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True) # sets warn_only to False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_enabled,
                                           warn_only=prev_warn_only)
