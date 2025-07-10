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
from tensorrt_bionemo._torch.layers.affinity import (AffinityModule,
                                                     compute_distogram,
                                                     create_cross_pair_mask)
from tensorrt_bionemo.configs import AffinityModuleConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.runtime.backend import BackendBase, BackendBuilder
from tensorrt_bionemo.runtime.misc import CUASSERT, ensure_contiguous


class AffinityModuleTorch(BackendBase):
    # Boltz2 affinity module
    IMPL_CLASS = AffinityModule

    def __init__(self,
                 config: AffinityModuleConfig,
                 load_weights_fn: Optional[Callable] = None,
                 impl: nn.Module = None):
        super().__init__(config, load_weights_fn, impl)

        triangle_metadata_cls = get_attention_backend(
            config.triangle_attn_backend).Metadata

        self.attn_metadatas = {
            "triangle_attn": triangle_metadata_cls(mapping=config.mapping),
        }

        boundaries = torch.linspace(2, config.max_dist,
                                    config.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)

    def forward(self,
                s_inputs: torch.Tensor,
                z: torch.Tensor,
                x_pred: torch.Tensor,
                feats: dict[str, torch.Tensor],
                multiplicity=1,
                **kwargs) -> dict[str, torch.Tensor]:
        # Sanity check
        assert multiplicity == 1, "Multiplicity > 1 is not supported"
        assert "token_to_rep_atom" in feats, "token_to_rep_atom is required"
        assert "pad_token_mask" in feats, "pad_token_mask is required"
        assert "mol_type" in feats, "mol_type is required"
        assert "affinity_token_mask" in feats, "affinity_token_mask is required"

        distogram = compute_distogram(x_pred, self.boundaries,
                                      feats["token_to_rep_atom"], multiplicity)
        cross_pair_mask_0, cross_pair_mask_1 = \
                    create_cross_pair_mask(
                                    feats["pad_token_mask"],
                                    feats["mol_type"],
                                    feats["affinity_token_mask"],
                                    multiplicity,
                                    include_mask_for_head=True)

        original_dtype = s_inputs.dtype
        # Cast the output to the expected dtype
        if self.config.triangle_attn_backend == "TRIFAST":
            self.attn_metadatas["triangle_attn"].closest_n = get_closest_n(
                s_inputs.shape[1] // self.config.mapping.dcp_size)
        pred_value, logits_binary = self._module(
            s_inputs.to(self.config.torch_dtype), z.to(self.config.torch_dtype),
            distogram.to(torch.int32),
            cross_pair_mask_0.to(self.config.torch_dtype),
            cross_pair_mask_1.to(self.config.torch_dtype))
        return {
            "affinity_pred_value": pred_value.to(original_dtype),
            "affinity_logits_binary": logits_binary.to(original_dtype)
        }


class AffinityModuleTRT(BackendBase):
    IMPL_CLASS = None

    def __init__(self,
                 config: AffinityModuleConfig,
                 load_weights_fn: Optional[Callable] = None,
                 impl: nn.Module = None):
        super().__init__(config, load_weights_fn, impl)
        self.trt_dtype = str_dtype_to_trt(config.dtype)
        self.int32_dtype = str_dtype_to_trt("int32")
        boundaries = torch.linspace(2, config.max_dist,
                                    config.num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)

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
            z_dims = self.engine.get_tensor_profile_shape("z", i)
            min_opt = z_dims[0]
            max_opt = z_dims[-1]

            min_s = min_opt[1]  # 0: batch_size, 1: seqlen
            max_s = max_opt[1]  # 0: batch_size, 1: seqlen

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

    @ensure_contiguous
    def forward(self,
                s_inputs: torch.Tensor,
                z: torch.Tensor,
                x_pred: torch.Tensor,
                feats: dict[str, torch.Tensor],
                multiplicity=1,
                **kwargs) -> dict[str, torch.Tensor]:
        # Ensure the inputs are contiguous
        self.switch_opt_profile(s_inputs.shape[1])
        original_dtype = s_inputs.dtype

        distogram = compute_distogram(x_pred, self.boundaries,
                                      feats["token_to_rep_atom"], multiplicity)
        cross_pair_mask_0, cross_pair_mask_1 = \
                    create_cross_pair_mask(
                                    feats["token_pad_mask"],
                                    feats["mol_type"],
                                    feats["affinity_token_mask"],
                                    multiplicity,
                                    include_mask_for_head=True)

        # TODO: Use config.get_input_names() to get the input names
        inputs = {
            "s": s_inputs.to(self.config.torch_dtype),
            "z": z.to(self.config.torch_dtype),
            "distogram": distogram.to(torch.int32),
            "cross_pair_mask_0": cross_pair_mask_0.to(self.config.torch_dtype),
            "cross_pair_mask_1": cross_pair_mask_1.to(self.config.torch_dtype)
        }

        output_info = self.session.infer_shapes([
            TensorInfo("s", dtype=self.trt_dtype, shape=s_inputs.shape),
            TensorInfo("z", dtype=self.trt_dtype, shape=z.shape),
            TensorInfo(
                "distogram", dtype=self.int32_dtype, shape=distogram.shape),
            TensorInfo("cross_pair_mask_0",
                       dtype=self.trt_dtype,
                       shape=cross_pair_mask_0.shape),
            TensorInfo("cross_pair_mask_1",
                       dtype=self.trt_dtype,
                       shape=cross_pair_mask_1.shape),
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
        pred_value = outputs["pred_value"].to(original_dtype)
        logits_binary = outputs["logits_binary"].to(original_dtype)
        return {
            "affinity_pred_value": pred_value,
            "affinity_logits_binary": logits_binary
        }


class AffinityModuleBackendBuilder(BackendBuilder):
    BACKEND_CLASSES = {"trt": AffinityModuleTRT, "torch": AffinityModuleTorch}
    CONFIG_CLASS = AffinityModuleConfig
