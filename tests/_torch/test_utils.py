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

"""Tests for the ``safe_generator`` context manager in ``_torch.utils``.

``safe_generator`` yields a private RNG seeded from the default generator's
current state, so its draws reproduce exactly what the default generator would
have produced. On *clean* exit it advances the default generator to match; if
the ``with`` body raises, the default generator is left untouched. These tests
pin that contract on CPU (and CUDA when available) without any model.
"""

import pytest
import torch

from tensorrt_bionemo._torch.utils import safe_generator

SEED = 1234

# safe_generator uses the CPU RNG path on "cpu" and the CUDA RNG path on
# "cuda", so exercise both when a GPU is present.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request) -> torch.device:
    return torch.device(request.param)


def _default_rng_state(device: torch.device) -> torch.Tensor:
    """Snapshot the *default* generator state for ``device``."""
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.get_rng_state()


def test_yields_generator_on_device(device):
    """The context manager yields a ``torch.Generator`` on the right device."""
    with safe_generator(device) as generator:
        assert isinstance(generator, torch.Generator)
        assert generator.device.type == device.type


def test_draws_match_the_unwrapped_default_stream(device):
    """Wrapped draws reproduce the values the default generator would give.

    Seeding the run identically, drawing from the yielded private generator
    must be bit-identical to drawing straight from the default generator.
    """
    torch.manual_seed(SEED)
    ref_a = torch.randn(4, device=device)
    ref_b = torch.randn(4, device=device)

    torch.manual_seed(SEED)
    with safe_generator(device) as generator:
        got_a = torch.randn(4, device=device, generator=generator)
        got_b = torch.randn(4, device=device, generator=generator)

    assert torch.equal(got_a, ref_a)
    assert torch.equal(got_b, ref_b)


def test_default_generator_untouched_inside_block(device):
    """Draws inside the block go to the private generator, not the default."""
    torch.manual_seed(SEED)
    before = _default_rng_state(device)
    with safe_generator(device) as generator:
        torch.randn(64, device=device, generator=generator)
        # Still inside the block: the default generator must not have moved.
        assert torch.equal(_default_rng_state(device), before)


def test_commit_advances_default_generator_on_clean_exit(device):
    """On clean exit the default generator is advanced to mirror the draws.

    A downstream default draw after the block must continue the stream exactly
    as if the wrapped draws had used the default generator directly.
    """
    torch.manual_seed(SEED)
    inside = torch.randn(4, device=device)   # what the wrapped draw consumes
    ref_after = torch.randn(4, device=device)  # downstream default draw

    torch.manual_seed(SEED)
    with safe_generator(device) as generator:
        got_inside = torch.randn(4, device=device, generator=generator)
    got_after = torch.randn(4, device=device)  # from the default generator

    assert torch.equal(got_inside, inside)
    assert torch.equal(got_after, ref_after)


def test_exception_leaves_default_generator_untouched(device):
    """If the body raises, the default generator state is not committed."""
    torch.manual_seed(SEED)
    before = _default_rng_state(device)

    with pytest.raises(RuntimeError, match="boom"):
        with safe_generator(device) as generator:
            torch.randn(64, device=device, generator=generator)
            raise RuntimeError("boom")

    assert torch.equal(_default_rng_state(device), before)


def test_seeds_from_current_state_not_a_fixed_seed(device):
    """The private generator tracks the default's *current* state each time.

    Two consecutive scopes, with default draws in between, must produce
    different streams — proving the generator is reseeded from the live default
    state rather than a constant.
    """
    torch.manual_seed(SEED)
    with safe_generator(device) as generator:
        first = torch.randn(4, device=device, generator=generator)
    torch.randn(4, device=device)  # advance the default generator
    with safe_generator(device) as generator:
        second = torch.randn(4, device=device, generator=generator)

    assert not torch.equal(first, second)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
