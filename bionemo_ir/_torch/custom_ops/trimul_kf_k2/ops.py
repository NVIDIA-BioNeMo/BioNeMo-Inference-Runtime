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
"""Tile selection and PyTorch reference for the SM90 TriMul KF K2."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from cuda.bindings import driver

from ._config import COUNTER_WORDS, TrimulKFK2Tile
from .cutedsl import TrimulKFK2CuTe, ab_layout_ok

_SUPPORTED_DTYPE = torch.bfloat16
#: Token counts must keep every matrix row 16-byte aligned for TMA.
TOKEN_ALIGN = 8
# One backend per SM version, so a process that drives GPUs of different SMs gets each its own configs.
_trimul_kf_k2_instances: dict[int, TrimulKFK2CuTe] = {}
# The dynamic schedule's work counters: (device, stream) for eager launches, (device, stream, capture id) under
# CUDA-graph capture, at most one of each per stream (see work_counter).
_work_counters: dict[tuple[int, ...], torch.Tensor] = {}


def work_counter(device: torch.device) -> torch.Tensor:
    """The work counter of a dynamic K2 launch on ``device``'s current stream: int32 ``[COUNTER_WORDS]``, zero.

    Every launch leaves its counter at zero, so the launches of one stream, which run in order, share one. A launch
    under CUDA-graph capture takes its capture's own counter instead, zeroed by a node ahead of the graph's first
    use of it, so the graph zeroes it on every replay and shares it with no other graph and no eager launch: graphs
    captured on different streams, or in different captures, replay concurrently without racing on it. Replaying
    one graph on two streams at once is unsupported, as it would share the counter, like its activations.

    A capture's counter is one of its graph's intermediates: the graph's memory pool keeps its block for every
    replay. The op holds it only while the capture can still launch K2, and lets it go when its stream starts
    another capture, since a stream captures one sequence at a time. Repeated captures thus hold one counter per
    stream rather than one per graph.
    """
    stream = torch.cuda.current_stream(device).cuda_stream
    key = (device.index, stream, *_capture_id(stream))
    counter = _work_counters.get(key)
    if counter is None:
        if len(key) > 2:
            for ended in [held for held in _work_counters if len(held) > 2 and held[:2] == key[:2]]:
                del _work_counters[ended]
        counter = _work_counters[key] = torch.zeros(COUNTER_WORDS, dtype=torch.int32, device=device)
    return counter


def _capture_id(stream: int) -> tuple[int, ...]:
    """``(id,)`` of the CUDA-graph capture ``stream`` is in, unique over the process, or ``()``."""
    result, status, capture_id, *_ = driver.cuStreamGetCaptureInfo(driver.CUstream(stream))
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamGetCaptureInfo failed: {result}")
    return (int(capture_id),) if status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE else ()


@dataclass(frozen=True)
class TrimulKFK2Op:
    """K2 bound to one hidden width and contraction direction; see :func:`get_trimul_kf_k2_op`.

    A tile on the dynamic schedule claims its work from the :func:`work_counter` of the stream it runs on.
    """

    backend: TrimulKFK2CuTe
    D: int
    outgoing: bool
    _tiles: dict[int, TrimulKFK2Tile] = field(default_factory=dict, init=False, repr=False, compare=False)

    def select(self, n: int) -> TrimulKFK2Tile:
        """The tile a call over ``n`` tokens runs."""
        tile = self._tiles.get(n)
        if tile is None:
            selected = self.backend.select(self.D, n, self.outgoing)
            if selected is None:
                raise ValueError(f"trimul KF K2 ships no config for D={self.D}")
            tile = self._tiles[n] = selected[1]
        return tile

    def accepts(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Whether a call on channel-major ``a`` and ``b`` ``[B, D, N, N]`` can run."""
        if a.dim() != 4 or a.dtype != _SUPPORTED_DTYPE or not a.is_cuda:
            return False
        B, D, N, N_k = a.shape
        # b shares a's strides, so a's layout check covers both; only b's base address is left to check
        return (
            B > 0
            and D == self.D
            and 0 < N == N_k
            and N % TOKEN_ALIGN == 0
            and b.shape == a.shape
            and b.dtype == _SUPPORTED_DTYPE
            and b.is_cuda
            and b.stride() == a.stride()
            and ab_layout_ok(a)
            and b.data_ptr() % 16 == 0
        )

    def __call__(self, a: torch.Tensor, b: torch.Tensor, pitched: bool = False) -> torch.Tensor:
        """The bf16 product ``[B, D, N, N]`` of K1's ``a`` and ``b``, for the tile tuned for ``N``.

        Args:
            a: bf16 ``[B, D, N, N]`` channel-major operand, dense or a view of K1's padded planes.
            b: ``a``'s partner, laid out the same way.
            pitched: Write the product's rows at ``a``'s row pitch ``P``, channels ``N`` rows apart: a
                ``[B, D, N, N]`` view of a ``[B, D, N, P]`` buffer, whose rows start on 128-byte lines when
                K1 pads its planes. Only a K3 that reads a pitched product takes it (``reads_pitch``).
                Dense otherwise.
        """
        if not self.accepts(a, b):
            raise ValueError(
                f"trimul KF K2 needs non-empty bf16 CUDA a and b [B, {self.D}, N, N] with N a multiple of "
                f"{TOKEN_ALIGN}, dense or K1's padded planes (same strides, unit column stride, row pitch and plane "
                f"stride multiples of 8 elements)"
            )
        N = a.shape[2]
        tile = self.select(N)
        counter = work_counter(a.device) if tile.dynamic else None
        return self.backend.run(a, b, self.outgoing, tile, a.stride(2) if pitched else N, counter)


def get_trimul_kf_k2_op(dtype: torch.dtype | None, hidden_dim: int, outgoing: bool) -> TrimulKFK2Op | None:
    """Return K2 when this build ships it for ``hidden_dim`` and the direction on the current GPU, else ``None``."""
    if dtype != _SUPPORTED_DTYPE or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    backend = _trimul_kf_k2_instances.get(sm_version)
    if backend is None:
        backend = _trimul_kf_k2_instances.setdefault(sm_version, TrimulKFK2CuTe(sm_version))
    if not backend.ships(hidden_dim, outgoing):
        return None
    return TrimulKFK2Op(backend, hidden_dim, outgoing)
