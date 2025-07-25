# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Callable, Optional

import torch
import torch.nn as nn
from tensorrt_llm._utils import str_dtype_to_trt

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.transformers import TokenTransformer
from tensorrt_bionemo.configs import TokenTransformerConfig
from tensorrt_bionemo.runtime.allocator import BaseContextMemoryManager
from tensorrt_bionemo.runtime.backend import BackendBase, BackendBuilder
from tensorrt_bionemo.runtime.misc import dtype_context, ensure_contiguous


class TokenTransformerTorch(BackendBase):
    IMPL_CLASS = TokenTransformer

    def __init__(self,
                 config: TokenTransformerConfig,
                 load_weights_fn: Optional[Callable] = None,
                 impl: nn.Module = None):
        super().__init__(config, load_weights_fn, impl)
        self.metadata_cls = get_attention_backend(
            self.config.pairwise_attn_backend).Metadata
        self.attn_metadata = self.metadata_cls(mapping=self.config.mapping,
                                               bias_cache={})

    def reset(self):
        # Use OrderedDict to maintain the order of the keys from layers
        self.attn_metadata.bias_cache = OrderedDict({})

    def gather_bias(self) -> torch.Tensor:
        if self.config.version == "v1":
            keys = list(self.attn_metadata.bias_cache.keys())
            biases = [self.attn_metadata.bias_cache[k] for k in keys]
            return torch.stack(biases, dim=-1).contiguous()
        raise NotImplementedError(
            "Gather bias is not implemented for version 2")

    def _forward_v1(self,
                    a: torch.Tensor,
                    s: torch.Tensor,
                    z: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=s.dtype) as cast_func:
            # TODO: add allreduce parameters here
            a = cast_func(self._module)(a,
                                        s,
                                        z=z,
                                        mask=mask,
                                        attn_metadata=self.attn_metadata)
        return a

    def _forward_v2(self,
                    a: torch.Tensor,
                    s: torch.Tensor,
                    bias: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        with dtype_context(expected_dtype=self.config.torch_dtype,
                           original_dtype=s.dtype) as cast_func:
            # TODO: add allreduce parameters here
            a = cast_func(self._module)(a,
                                        s,
                                        z=bias,
                                        mask=mask,
                                        attn_metadata=self.attn_metadata)
        return a

    def forward(self, *args, **kwargs):
        if self.config.version == "v1":
            return self._forward_v1(*args, **kwargs)
        elif self.config.version == "v2":
            return self._forward_v2(*args, **kwargs)
        else:
            raise ValueError(
                f"Invalid token transformer version: {self.config.version}")


class TokenTransformerTRT(BackendBase):
    IMPL_CLASS = None

    def __init__(self,
                 config: TokenTransformerConfig,
                 load_weights_fn: Optional[Callable] = None,
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        super().__init__(config,
                         load_weights_fn,
                         impl,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)
        self._concat_bias_cache: torch.Tensor = None  # for Boltz-1 model

    def load_weights(self,
                     checkpoint_dir: str,
                     world_size: int,
                     rank: int,
                     context_without_device_memory: bool = True,
                     address=None,
                     stream=None,
                     torch_load_weights_fn: Optional[Callable] = None,
                     torch_local_checkpoint: str = None,
                     **kwargs):
        """
        Load the token transformer engine from the checkpoint directory.
        Args:
            checkpoint_dir: The directory containing the token transformer engine.
            world_size: The world size of the engine.
            rank: The rank of the engine.
            context_without_device_memory: Whether to create a context without device memory.
            address: The address of the device memory.
            stream: The stream to use for the engine.
            torch_load_weights_fn:
                The function to load the weights for the token transformer torch backend.
                This is only used for the Boltz-1 model. For the first iteration to compute the biases
            torch_local_checkpoint:
                The local checkpoint for the token transformer torch backend.
                This is only used for the Boltz-1 model.
        """
        # Set attributes for the allocator
        self._checkpoint_dir = checkpoint_dir
        self._world_size = world_size
        self._runtime_rank = rank

        # Store the custom stream
        self._custom_stream = stream

        # Delegate engine management to the allocator
        if self._context_memory_allocator is not None:
            self._context_memory_allocator.add_handle(self, stream=stream)

        if self.config.version == "v1":  # Boltz-1 model
            if torch_load_weights_fn is not None:
                self._torch_module = TokenTransformerTorch(
                    self.config, torch_load_weights_fn)
                self._torch_module.load_weights(torch_local_checkpoint,
                                                world_size,
                                                rank,
                                                compile=True)

    def reset(self):
        if self.config.version == "v1":
            # Delete the bias cache
            self._torch_module.reset()
            self._concat_bias_cache = None

    def _forward_v1(self,
                    a: torch.Tensor,
                    s: torch.Tensor,
                    z: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        if self._concat_bias_cache is None:
            # Run the torch module backend only once to compute the biases
            a = self._torch_module(a, s, z, mask, **kwargs)
            self._concat_bias_cache = self._torch_module.gather_bias()
            return a
        return self._forward_internal(a, s, self._concat_bias_cache, mask,
                                      **kwargs)

    def _forward_v2(self,
                    a: torch.Tensor,
                    s: torch.Tensor,
                    bias: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    **kwargs) -> torch.Tensor:
        return self._forward_internal(a, s, bias, mask, **kwargs)

    @ensure_contiguous
    def _forward_internal(self,
                          a: torch.Tensor,
                          s: torch.Tensor,
                          z: torch.Tensor = None,
                          mask: torch.Tensor = None,
                          **kwargs) -> torch.Tensor:
        original_dtype = s.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "a": a.to(self.config.torch_dtype),
            "s": s.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "mask": mask.to(self.config.torch_dtype)
        }

        # Use the allocator from the base class for execution
        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        return outputs["output_a"].to(original_dtype)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.config.version == "v1":
            return self._forward_v1(*args, **kwargs)
        elif self.config.version == "v2":
            return self._forward_v2(*args, **kwargs)
        else:
            raise ValueError(
                f"Invalid token transformer version: {self.config.version}")


class TokenTransformerBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {
        "torch": TokenTransformerTorch,
        "trt": TokenTransformerTRT
    }
    CONFIG_CLASS = TokenTransformerConfig
