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
"""Low-level execution support for dynamic-shape cuDNN operation graphs."""

from __future__ import annotations

import weakref
from collections.abc import Sequence

import cudnn
import torch

type TensorShape = tuple[int, ...]
type TensorStride = tuple[int, ...]


def _cudnn_data_type(dtype: torch.dtype) -> cudnn.data_type:
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.float32:
        return cudnn.data_type.FLOAT
    raise ValueError(f"cuDNN operation graphs do not support output dtype {dtype}")


class DynamicGraph:
    """A built cuDNN graph that accepts runtime shape overrides."""

    def __init__(self, device: torch.device, dtype: torch.dtype, kernel_cache: object) -> None:
        if device.type != "cuda":
            raise ValueError(f"dynamic cuDNN graphs require a CUDA device, got {device}")
        self.device = device
        self.dtype = dtype
        self._workspace_size: int | None = None
        self._workspace: torch.Tensor | None = None
        with torch.cuda.device(device):
            self._handle = cudnn.create_handle()
            self.graph = cudnn.pygraph(
                handle=self._handle,
                io_data_type=_cudnn_data_type(dtype),
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
                kernel_cache=kernel_cache,
                is_dynamic_shape_enabled=True,
                is_override_shape_enabled=True,
            )
        self._handle_finalizer = weakref.finalize(self, cudnn.destroy_handle, self._handle)

    def tensor(
        self,
        shape: TensorShape,
        stride: TensorStride,
        name: str,
    ) -> cudnn.tensor:
        """Create a concrete I/O tensor descriptor.

        Args:
            shape: Cache shape used to build the dynamic execution plan.
            stride: Cache-shape strides in elements.
            name: Graph-local tensor name.

        Returns:
            The cuDNN tensor descriptor.
        """
        return self.graph.tensor(
            dim=list(shape),
            stride=list(stride),
            data_type=_cudnn_data_type(self.dtype),
            name=name,
        )

    def build(self) -> None:
        """Validate the graph and build its selected execution plan."""
        with torch.cuda.device(self.device):
            self.graph.validate()
            self.graph.build_operation_graph()
            self.graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            self.graph.check_support()
            self.graph.build_plans()
            self._workspace_size = self.graph.get_workspace_size()
            if self._workspace_size == 0:
                self._workspace = torch.empty(0, device=self.device, dtype=torch.uint8)

    def execute(
        self,
        bindings: dict[cudnn.tensor, torch.Tensor],
        override_uids: Sequence[int],
        override_shapes: Sequence[TensorShape],
        override_strides: Sequence[TensorStride],
    ) -> bool:
        """Execute plan zero with runtime dimensions and strides.

        Args:
            bindings: Concrete tensors bound to graph I/O descriptors.
            override_uids: Dynamic tensor descriptor identifiers.
            override_shapes: Runtime shape for each dynamic descriptor.
            override_strides: Runtime strides for each dynamic descriptor.

        Returns:
            Whether cuDNN accepted the runtime shape.
        """
        if self._workspace_size is None:
            raise RuntimeError("the dynamic cuDNN graph must be built before execution")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            cudnn.set_stream(self._handle, stream.cuda_stream)
            workspace = self._workspace
            temporary_workspace = workspace is None
            if workspace is None:
                workspace = torch.empty(self._workspace_size, device=self.device, dtype=torch.uint8)
            try:
                self.graph.execute_plan_at_index(
                    bindings,
                    workspace,
                    0,
                    handle=self._handle,
                    override_uids=override_uids,
                    override_shapes=override_shapes,
                    override_strides=override_strides,
                )
            except RuntimeError as error:
                if "CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE" in str(error):
                    return False
                raise
            if temporary_workspace:
                workspace.record_stream(stream)
            return True
