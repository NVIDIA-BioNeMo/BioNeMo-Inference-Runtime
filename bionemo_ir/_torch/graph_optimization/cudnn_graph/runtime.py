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
"""Runtime eligibility checks for cuDNN operation graphs."""

from __future__ import annotations

from collections.abc import Callable, Collection

import torch
import torch.nn as nn


class CudnnGraphModule(nn.Module):
    """Module base that prepares cuDNN plans after device or dtype moves."""

    def __getstate__(self) -> dict[str, object]:
        """Exclude process-local cuDNN handles from serialized module state."""
        state = super().__getstate__()
        if "_cudnn_graph_plans" in state:
            state["_cudnn_graph_plans"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore module state and rebuild eligible process-local plans."""
        super().__setstate__(state)
        self._prepare_cudnn_graphs()

    def _apply(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> CudnnGraphModule:
        """Apply a tensor transform, then prepare plans on the new device."""
        super()._apply(function, recurse=recurse)
        self._prepare_cudnn_graphs()
        return self

    def _prepare_cudnn_graphs(self) -> None:
        """Prepare operation graphs whose dimensions are known by the module."""
        raise NotImplementedError


def can_use_cudnn_graph(
    tensor: torch.Tensor,
    *,
    enabled: bool,
    dtypes: Collection[torch.dtype] = (torch.bfloat16,),
    require_batch_one: bool = False,
) -> bool:
    """Return whether a tensor can enter a configured cuDNN graph path.

    Args:
        tensor: Candidate operation-graph input.
        enabled: Module-level opt-in flag.
        dtypes: Accepted input dtypes.
        require_batch_one: Require a leading dimension of one.

    Returns:
        Whether the tensor meets all configured constraints.
    """
    return (
        enabled
        and tensor.device.type == "cuda"
        and tensor.dtype in dtypes
        and (not require_batch_one or (tensor.ndim > 0 and tensor.shape[0] == 1))
    )
