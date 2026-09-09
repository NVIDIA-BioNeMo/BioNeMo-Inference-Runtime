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
"""Tests for :mod:`tests.common.test_utils.openfold3.batched_input_tools`."""

import pytest
import torch

from tests.common.test_utils.openfold3.batched_input_tools import (
    AVAILABILITY_EXC,
    harness_skip_reason,
    make_batched_diffusion_inputs,
)


def test_make_batched_diffusion_inputs():
    """The assembled B=2 batch runs through the real DiffusionModule."""
    sample_ids = ("T1038", "T1047s1")
    reason = harness_skip_reason(sample_ids)
    if reason is not None:
        pytest.skip(reason)

    try:
        module, batched = make_batched_diffusion_inputs(sample_ids)
    except AVAILABILITY_EXC as exc:
        pytest.skip(f"openfold3 weights/metadata unavailable ({type(exc).__name__}: {exc})")

    assert batched["xl_noisy"].shape[0] == len(sample_ids)
    with torch.no_grad():
        out = module(**batched)
    assert out.shape[0] == len(sample_ids) and out.shape[-1] == 3
    assert torch.isfinite(out).all()
