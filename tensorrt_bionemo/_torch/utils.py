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

import contextlib
from collections.abc import Callable

import torch
from torch import nn


def make_graph_safe_generator(device: torch.device) -> torch.Generator:
    """Return a private RNG seeded from the current default-generator state.

    CUDA-graph capture registers the *default* CUDA generator with the captured
    graph; any subsequent **eager** RNG drawn from the default generator then
    raises "Offset increment outside graph capture encountered unexpectedly".
    Drawing eager randomness from a private generator — one that is never
    registered with any graph — sidesteps this entirely.

    The private generator is seeded by cloning the default generator's current
    state, so it reproduces the exact RNG stream the default generator would
    have produced. Numerics (and run-to-run / eager-vs-graph parity) are
    therefore unchanged; only the generator object backing the draws differs.
    """
    device = torch.device(device)
    generator = torch.Generator(device=device)
    if device.type == "cuda":
        generator.set_state(torch.cuda.get_rng_state(device))
    else:
        generator.set_state(torch.get_rng_state())
    return generator


def commit_graph_safe_generator(generator: torch.Generator, device: torch.device) -> None:
    """Advance the default generator to match a private generator's state.

    Pairs with :func:`make_graph_safe_generator`: after drawing from the
    private generator, copy its (now advanced) state back into the default
    generator so downstream RNG consumers observe exactly the state progression
    they would have if the draws had used the default generator directly. This
    keeps numerics identical to the un-wrapped model while ensuring no eager
    draw ever touches the graph-registered default generator.
    """
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_rng_state(generator.get_state(), device)
    else:
        torch.set_rng_state(generator.get_state())


@contextlib.contextmanager
def safe_generator(device: torch.device):
    """Scope a graph-safe private RNG, committing its state back on exit.

    Combines :func:`make_graph_safe_generator` and
    :func:`commit_graph_safe_generator`: yields a private generator seeded from
    the default generator's state, and on **clean** exit advances the default
    generator to match. If the ``with`` body raises, the default generator is
    left untouched (mirroring the un-wrapped code, which only committed after
    the rollout completed). Draw all of a graph-captured region's eager
    randomness from the yielded generator::

        with safe_generator(device) as generator:
            x = torch.randn(..., generator=generator)
    """
    generator = make_graph_safe_generator(device)
    yield generator
    commit_graph_safe_generator(generator, device)


def recursive_calling_load_weights(module: nn.Module, weights: dict, filter_func: Callable = None) -> set[str]:
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

    for name, submodule in module.named_modules():
        if filter_func is not None and filter_func(name, submodule):
            continue
        if len(submodule._parameters) > 0:
            try:
                if hasattr(submodule, "load_weights"):
                    submodule.load_weights(weights=weights[name])
                else:
                    module_weights = weights[name][0]
                    for n, p in submodule._parameters.items():
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
    torch.use_deterministic_algorithms(True)  # sets warn_only to False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)
