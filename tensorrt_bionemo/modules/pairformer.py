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
from typing import Callable, Optional

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from cuda import cudart
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime import Session, TensorInfo
from tensorrt_llm.runtime.session import _scoped_stream

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.transformers import PairformerModule
from tensorrt_bionemo.confs.modules.transformers import PairformerConfig
from tensorrt_bionemo.mapping import Mapping


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


@torch.compiler.disable
def get_closest_n(s):
    return 2**int(np.ceil(np.log2(s.shape[1])))


class PairformerTorch(nn.Module):

    def __init__(self,
                 config: PairformerConfig,
                 load_weights_fn: Optional[Callable] = None):
        super().__init__()
        self.config = config

        self._load_weights_fn = load_weights_fn
        pairwise_metadata_cls = get_attention_backend(
            config.pairwise_attn_backend).Metadata
        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadatas = {
            "triangle_attn": triangle_metadata_cls(mapping=config.mapping),
            "pairwise_attn": pairwise_metadata_cls(mapping=config.mapping),
        }

    def load_weights(self, checkpoint_dir: str, world_size: int, rank: int,
                     weights: dict, **kwargs):
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size
        self.runtime_rank = rank
        _module = PairformerModule(self.config)
        if self._load_weights_fn is not None:
            self._load_weights_fn(_module,
                                  checkpoint_dir=checkpoint_dir,
                                  world_size=world_size,
                                  rank=rank,
                                  weights=weights,
                                  **kwargs)
        _module.cuda()
        _module.eval()
        # TODO: Whether use torch.compile() or not, checking chunking configurations
        mode = None
        if self.config.mapping.world_size > 1:
            mode = "max-autotune-no-cudagraphs"
        self._module = torch.compile(_module,
                                     fullgraph=True,
                                     dynamic=True,
                                     mode=mode)

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = s.dtype
        if self.config.torch_dtype != s.dtype:
            s = s.to(self.config.torch_dtype)
            z = z.to(self.config.torch_dtype)
            mask = mask.to(self.config.torch_dtype)
            pair_mask = pair_mask.to(self.config.torch_dtype)
        if self.config.triangle_attn_backend == "TRIFAST":
            self.attn_metadatas["triangle_attn"].closest_n = get_closest_n(
                s // self.config.mapping.dcp_size)
        s, z = self._module(s,
                            z,
                            mask,
                            pair_mask,
                            attn_metadatas=self.attn_metadatas)
        if s.dtype != original_dtype:
            s = s.to(original_dtype)
            z = z.to(original_dtype)
        return s, z


class PairformerTRT(nn.Module):

    def __init__(self,
                 config: PairformerConfig,
                 load_weights_fn: Optional[Callable] = None):
        super().__init__()
        self.config = config
        self.trt_dtype = str_dtype_to_trt(config.dtype)
        self._load_weights_fn = load_weights_fn

    def load_weights(self,
                     checkpoint_dir: str,
                     world_size: int,
                     rank: int,
                     context_without_device_memory: bool = True,
                     address=None,
                     stream=None,
                     **kwargs):
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
            self.engine.get_tensor_profile_shape("s", i)
            min_opt = mask_dims[0]
            max_opt = mask_dims[-1]

            if self.config.support_batch:
                min_s = min_opt[1]  # 0: batch_size, 1: seqlen
                max_s = max_opt[1]  # 0: batch_size, 1: seqlen
            else:
                min_s = min_opt[0]  # 0: seqlen
                max_s = max_opt[0]  # 0: seqlen

            self.opt_profile_map[(min_s, max_s)] = i
        self.curr_profile = 0

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

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the inputs are contiguous
        self.switch_opt_profile(s.shape[1])
        if not self.config.support_batch:
            s = s.squeeze(0)
            z = z.squeeze(0)
            mask = mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)
        if not s.is_contiguous():
            s = s.contiguous()
        if not z.is_contiguous():
            z = z.contiguous()
        if not mask.is_contiguous():
            mask = mask.contiguous()
        if not pair_mask.is_contiguous():
            pair_mask = pair_mask.contiguous()
        original_dtype = s.dtype

        inputs = {
            "s": s.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "mask": mask.to(self.config.torch_dtype),
            "pair_mask": pair_mask.to(self.config.torch_dtype)
        }

        output_info = self.session.infer_shapes([
            TensorInfo("s", dtype=self.trt_dtype, shape=s.shape),
            TensorInfo("z", dtype=self.trt_dtype, shape=z.shape),
            TensorInfo("mask", dtype=self.trt_dtype, shape=mask.shape),
            TensorInfo("pair_mask", dtype=self.trt_dtype,
                       shape=pair_mask.shape),
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
        s = outputs["output_s"].to(original_dtype)
        z = outputs["output_z"].to(original_dtype)
        if not self.config.support_batch:
            s = s.unsqueeze(0)
            z = z.unsqueeze(0)
        torch.save(s, "trt_s.pt")
        torch.save(z, "trt_z.pt")
        return s, z


class PairformerBackendBuilder:
    BACKEND_CLASSES = {"trt": PairformerTRT, "torch": PairformerTorch}

    @staticmethod
    def build(config: PairformerConfig,
              checkpoint_dir: str = None,
              world_size: int = 1,
              rank: int = 0,
              weights: dict = None,
              load_weights_fn: Optional[Callable] = None,
              **kwargs) -> nn.Module:
        """
        Build Pairformer module based on the backend type.

        Args:
            config: The configuration for the Pairformer module.
            checkpoint_dir: The directory to load the checkpoint from.
            world_size: The number of processes to use.
            rank: The rank of the process.
            weights: The weights to load into the module.
                If None, the weights will be loaded from the checkpoint directory.
            load_weights_fn: The function to load the weights.
                If provided, the weights will be loaded by this function, if not, the method `load()` of the backend class will be called.
        """
        backend = config.backend
        if backend not in PairformerBackendBuilder.BACKEND_CLASSES:
            raise ValueError(f"Invalid backend: {backend}")
        module = PairformerBackendBuilder.BACKEND_CLASSES[backend](
            config, load_weights_fn)

        module.load_weights(checkpoint_dir=checkpoint_dir,
                            world_size=world_size,
                            rank=rank,
                            weights=weights,
                            **kwargs)
        return module
