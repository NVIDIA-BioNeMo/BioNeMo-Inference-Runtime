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
import os
from collections import OrderedDict
from typing import Callable, Optional

import tensorrt as trt
import torch
import torch.nn as nn
from cuda import cudart
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import Session, TensorInfo
from tensorrt_llm.runtime.session import _scoped_stream

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.transformers import TokenTransformer
from tensorrt_bionemo.configs import TokenTransformerConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.runtime.backend import BackendBase, BackendBuilder
from tensorrt_bionemo.runtime.misc import (CUASSERT, dtype_context,
                                           ensure_contiguous)


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
                 impl: nn.Module = None):
        super().__init__(config, load_weights_fn, impl)
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
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size
        self.runtime_rank = rank
        if self._load_weights_fn is not None:
            self._load_weights_fn(
                self,
                checkpoint_dir=checkpoint_dir,
                world_size=world_size,
                rank=rank,
                context_without_device_memory=context_without_device_memory,
                address=address,
                stream=stream,
                **kwargs)
            return

        assert self.config is not None
        config_dtype = self.config.dtype
        logger.info(f"Engine dtype: {config_dtype}")
        self.disable_custom_all_reduce = self.config.disable_custom_all_reduce
        self.tp_size = self.config.mapping.tp_size
        self.dcp_size = self.config.mapping.dcp_size
        assert world_size == self.world_size, \
            (f'Engine world size ({world_size}) != Runtime world size ({self.world_size})')
        self.engine_name = f"rank{self.runtime_rank}.engine"
        self.runtime_mapping = Mapping(world_size=self.world_size,
                                       rank=self.runtime_rank,
                                       tp_size=self.tp_size,
                                       dcp_size=self.dcp_size)
        if self.world_size > 1 and not self.disable_custom_all_reduce:
            # init_all_reduce_helper()
            _, self.workspace = CustomAllReduceHelper.allocate_workspace(
                self.runtime_mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.runtime_mapping.tp_size))
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.current_stream().cuda_stream
        self.serialize_path = os.path.join(self.checkpoint_dir,
                                           self.engine_name)
        with open(self.serialize_path, 'rb') as f:
            engine_buffer = f.read()
            assert engine_buffer is not None
        logger.info(f"Deserialize engine from {self.serialize_path}")
        self.runtime = trt.Runtime(logger.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        self.device_memory_size = self.engine.device_memory_size_v2
        self.address = None

        if not context_without_device_memory:
            self.context = self.engine.create_execution_context()
            with _scoped_stream() as stream:
                self.context.set_optimization_profile_async(0, stream)
        else:
            self.context = self.engine.create_execution_context_without_device_memory(
            )
            if address is None:
                address = CUASSERT(cudart.cudaMalloc(
                    self.device_memory_size))[0]
            self.context.set_device_memory(address, self.device_memory_size)
            self.address = address
            with _scoped_stream() as stream:
                self.context.set_optimization_profile_async(0, stream)
        # Initialize session
        self.session = Session()
        self.session._runtime = self.runtime
        self.session._context = self.context
        self.session.engine = self.engine

        self.session._print_engine_info()
        self.engine = self.session.engine
        logger.info(
            f"The memory required by the largest profile: {self.engine.device_memory_size_v2}"
        )
        self.context = self.session.context

        self.opt_profile_map = {}
        num_optimization_profiles = self.engine.num_optimization_profiles
        for i in range(num_optimization_profiles):
            mask_dims = self.engine.get_tensor_profile_shape("mask", i)
            min_opt = mask_dims[0]
            max_opt = mask_dims[-1]

            min_s = min_opt[1]  # 0: batch_size, 1: seqlen
            max_s = max_opt[1]  # 0: batch_size, 1: seqlen

            self.opt_profile_map[(min_s, max_s)] = i
        self.curr_profile = 0
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

    def get_backend_workspace(self):
        """ Return the address and the size of the workspace for the backend."""
        return self.address, self.device_memory_size

    def switch_opt_profile(self, input_length: int):
        found_profile = -1
        for k, v in self.opt_profile_map.items():
            if k[0] <= input_length <= k[1]:
                found_profile = v
        if found_profile == -1:
            raise ValueError(
                f"No suitable optimization profile found for the current rank. "
                f"Please check the engine configuration.")
        if found_profile != self.curr_profile:
            self.curr_profile = found_profile
            with _scoped_stream() as stream:
                self.context.set_optimization_profile_async(
                    self.curr_profile, stream)

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
        self.switch_opt_profile(a.shape[1])
        original_dtype = s.dtype

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "a": a.to(self.config.torch_dtype),
            "s": s.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "mask": mask.to(self.config.torch_dtype)
        }

        output_info = self.session.infer_shapes([
            TensorInfo("a", dtype=self.trt_dtype, shape=a.shape),
            TensorInfo("s", dtype=self.trt_dtype, shape=s.shape),
            TensorInfo("z", dtype=self.trt_dtype, shape=z.shape),
            TensorInfo("mask", dtype=self.trt_dtype, shape=mask.shape),
        ], self.context)
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in output_info
        }
        if self.world_size > 1 and not self.disable_custom_all_reduce:
            inputs["all_reduce_workspace"] = self.workspace
        ok = self.session.run(inputs,
                              outputs,
                              self.stream,
                              context=self.context)
        assert ok, "Runtime execution failed"
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
