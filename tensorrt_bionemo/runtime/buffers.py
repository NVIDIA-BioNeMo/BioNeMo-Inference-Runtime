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
"""Pre-allocated GPU buffer management for reducing allocation overhead.

Provides a lightweight ``dict``-based buffer pool that layers can share
across a forward pass.  Each buffer is lazily allocated on first use and
reused on subsequent calls as long as the requested shape / dtype match.

Usage at the model level::

    buffers: PreallocatedBuffers = {}
    for layer in self.layers:
        a = layer(a, s, z, ..., buffers=buffers)

Usage inside a layer::

    from tensorrt_bionemo.runtime.buffers import ensure_buffer

    attn_buf = ensure_buffer(
        buffers, "attn_output", (B, Sq, H, D), dtype, device)
    mha_o = self.attn.forward(q, k, v, biases=biases, output=attn_buf)
"""

import torch

PreallocatedBuffers = dict[str, torch.Tensor]


def ensure_buffer(
    buffers: PreallocatedBuffers | None,
    key: str,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a buffer for *key*, allocating only when shape/dtype change.

    When *buffers* is ``None`` the function returns ``None``, letting the
    caller fall back to its default allocation path.  This keeps every
    call-site a single line::

        buf = ensure_buffer(buffers, "attn_output", shape, dtype, device)
        # buf is either a ready-to-write tensor or None

    Args:
        buffers: Shared buffer dict (may be ``None``).
        key: Lookup key that uniquely identifies this allocation site.
        shape: Desired tensor shape.
        dtype: Desired tensor dtype.
        device: Desired tensor device.

    Returns:
        A tensor of the requested shape/dtype/device, or ``None`` if
        *buffers* is ``None``.
    """
    if buffers is None:
        return None
    buf = buffers.get(key)
    if buf is not None and buf.shape == shape and buf.dtype == dtype and buf.device == device:
        return buf
    buf = torch.empty(shape, dtype=dtype, device=device)
    buffers[key] = buf
    return buf
