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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import tensorrt as trt
import torch
from tensorrt_llm_lite._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm_lite.logger import logger
from tensorrt_llm_lite.runtime.session import (Session, TensorInfo,
                                               _scoped_stream)

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping

if TYPE_CHECKING:
    from .backend import BackendBase


class TRTEngineHandle:

    def __init__(self,
                 config: BaseConfig = None,
                 engine_dir: Union[Path, str] = None,
                 world_size: int = None,
                 rank: int = None,
                 stream: Any = None):
        """
        Initialize the TRTEngineHandle.
        TODO: Support cuda-graph for engine execution.
        Args:
            config: The config of the engine.
            engine_path: The path to the serialized engine.
            world_size: The world size of the engine.
            rank: The rank of the engine.
            stream: CUDA stream to use for the engine.
        """
        self._config = config
        self._runtime = None
        self._engine = None
        self._buffer_size = 0
        self._address = None
        self._context = None
        self._session = None
        self._world_size = world_size
        self._workspace = None
        self._rank = rank
        self._stream = stream
        self._engine_dir = engine_dir
        if self._stream is None:
            self._stream = torch.cuda.current_stream()
        self._tp_size = self._config.mapping.tp_size
        self._dcp_size = self._config.mapping.dcp_size

        config_dtype = self._config.dtype
        logger.info(f"Engine dtype: {config_dtype}")
        self._disable_custom_all_reduce = self._config.disable_custom_all_reduce
        assert world_size == self._world_size, \
            (f'Engine world size ({world_size}) != Runtime world size ({self._world_size})')
        self._engine_name = f"rank{self._rank}.engine"
        self._runtime_mapping = Mapping(world_size=self._world_size,
                                        rank=self._rank,
                                        tp_size=self._tp_size,
                                        dcp_size=self._dcp_size)

        self._serialize_path = os.path.join(engine_dir, self._engine_name)
        self._self_allocated = False
        self._using_torch_caching_allocator = True

    def get_stream(self):
        return self._stream

    def set_context_memory(self,
                           address: Any = None,
                           size_in_bytes: int = None):
        assert self._context is not None
        self._address = address
        self._buffer_size = size_in_bytes
        if self._address is not None:
            if self._buffer_size < self._engine.device_memory_size_v2:
                raise ValueError(
                    f"The size of the buffer must be greater than or equal to the size of the context memory. "
                    f"The size of the context memory is {self._engine.device_memory_size_v2}, but the size of the buffer is {size_in_bytes}."
                )
            self._context.set_device_memory(
                address, self._buffer_size)  # Maybe dangling pointer here

    @property
    def engine_dir(self):
        return self._engine_dir

    @property
    def device_memory_size(self):
        return self._engine.device_memory_size_v2

    @property
    def device_memory_address(self):
        return self._address

    def deserialize_engine(
        self,
        create_without_device_memory: bool = False,
        do_allocate_context_memory: bool = True,
    ):
        """
        Deserialize the engine from the serialized file.
        If do_allocate_context_memory is True, allocate the context memory for the engine.
        If do_allocate_context_memory is False, the context memory must to be set by set_context_memory.
        """
        with open(self._serialize_path, 'rb') as f:
            engine_buffer = f.read()
            assert engine_buffer is not None
        logger.info(f"Deserialize engine from {self._serialize_path}")
        self._runtime = trt.Runtime(logger.trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_buffer)
        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine from {self._serialize_path}")
        self._buffer_size = self._engine.device_memory_size_v2
        self._self_allocated = False
        if not create_without_device_memory:
            self._context = self._engine.create_execution_context()
            with _scoped_stream() as stream:
                self._context.set_optimization_profile_async(0, stream)
            self._self_allocated = True
            self._using_torch_caching_allocator = False
        else:
            self._context = self._engine.create_execution_context_without_device_memory(
            )
            if do_allocate_context_memory:
                # address = CUASSERT(cudart.cudaMalloc(self._buffer_size))[0]
                address = torch.cuda.caching_allocator_alloc(
                    self._buffer_size, stream=self._stream)
                self.set_context_memory(address, self._buffer_size)
                self._self_allocated = True
                self._using_torch_caching_allocator = True
            with _scoped_stream() as stream:
                self._context.set_optimization_profile_async(0, stream)

        # Initialize session
        self._session = Session()
        self._session._runtime = self._runtime
        self._session._context = self._context
        self._session.engine = self._engine

    def cleanup(self):
        if self._self_allocated and self._using_torch_caching_allocator:
            torch.cuda.caching_allocator_delete(self._address)
            self._address = None

    def __del__(self):
        self.cleanup()


class BaseContextMemoryManager:

    def __init__(self, auto_load: bool = True):
        """
        Initialize the BaseContextMemoryManager.
        Args:
            auto_load: Whether to automatically load the engines when adding a handle.
        """
        self._handles = dict()
        self._deserialized_handles = dict()
        self._auto_load = auto_load

    def remove_handle(self, backend_impl: 'BackendBase'):
        """
        Remove a TRTEngineHandle from the allocator.
        TODO: thread-safe
        """
        if backend_impl in self._handles:
            handle = self._handles[backend_impl]
            handle.cleanup()
            del self._handles[backend_impl]

    def add_handle(self,
                   backend_impl: 'BackendBase' = None,
                   stream: Any = None):
        """
        Add a TRTEngineHandle to the allocator.
        """
        handle = TRTEngineHandle(backend_impl.config,
                                 backend_impl.checkpoint_dir,
                                 backend_impl.world_size,
                                 backend_impl.runtime_rank, stream)
        self._handles[backend_impl] = handle
        if self._auto_load:
            self.load()

    def get_handles(self) -> dict['BackendBase', TRTEngineHandle]:
        """Get the TRT engine handles dictionary."""
        return self._handles

    def get_deserialized_handles(self) -> dict['BackendBase', TRTEngineHandle]:
        """Get the list of TRT engine handles that have allocated memory."""
        return self._deserialized_handles

    def add_deserialized_handle(self, backend_impl: 'BackendBase',
                                handle: TRTEngineHandle):
        """Add a TRT engine handle to the loaded handles dictionary."""
        self._deserialized_handles[backend_impl] = handle

    def build_opt_profile_map(self, handle: TRTEngineHandle):
        """ Build comprehensive optimization profile map for the given engine handle. Maps each input tensor to its optimization profiles with shape constraints """
        if not hasattr(handle, '_opt_profile_map'):
            handle._opt_profile_map = {}
            handle._curr_profile = 0

            num_optimization_profiles = handle._engine.num_optimization_profiles
            logger.info(
                f"Building optimization profile map for engine with {num_optimization_profiles} profiles"
            )

            # Get all input tensor names from the engine
            input_tensor_names = []
            for i in range(handle._engine.num_io_tensors):
                tensor_name = handle._engine.get_tensor_name(i)
                if handle._engine.get_tensor_mode(
                        tensor_name) == trt.TensorIOMode.INPUT:
                    input_tensor_names.append(tensor_name)

            if not input_tensor_names:
                logger.warning("No input tensors found in engine")
                return

            # Build profile map for each input tensor
            for tensor_name in input_tensor_names:
                tensor_profiles = []

                for profile_idx in range(num_optimization_profiles):
                    try:
                        tensor_dims = handle._engine.get_tensor_profile_shape(
                            tensor_name, profile_idx)
                        if tensor_dims:
                            # Store the min and max shapes for this profile
                            min_shape = tensor_dims[0]
                            max_shape = tensor_dims[-1]
                            tensor_profiles.append({
                                'profile_idx': profile_idx,
                                'min_shape': min_shape,
                                'max_shape': max_shape
                            })
                    except Exception as e:
                        logger.debug(
                            f"Could not get profile shape for tensor '{tensor_name}' profile {profile_idx}: {e}"
                        )
                        continue

                if tensor_profiles:
                    handle._opt_profile_map[tensor_name] = tensor_profiles
                    logger.debug(
                        f"Built {len(tensor_profiles)} profiles for tensor '{tensor_name}'"
                    )
                else:
                    logger.warning(
                        f"No valid profiles found for tensor '{tensor_name}'")

            logger.info(
                f"Built profile map with {len(handle._opt_profile_map)} input tensors"
            )

    def _get_opt_profiles(self, handle: TRTEngineHandle, tensor_name: str):
        """ Get optimization profiles for a specific tensor (cached for reuse) """
        if not hasattr(handle, '_opt_profile_map'):
            self.build_opt_profile_map(handle)

        return handle._opt_profile_map.get(tensor_name, [])

    def _shape_matches_profile(self, input_shape: tuple, profile_min: tuple,
                               profile_max: tuple) -> bool:
        """ Check if input shape matches the profile constraints """
        if len(input_shape) != len(profile_min) or len(input_shape) != len(
                profile_max):
            return False

        for i, (input_dim, min_dim, max_dim) in enumerate(
                zip(input_shape, profile_min, profile_max)):
            if not (min_dim <= input_dim <= max_dim):
                return False

        return True

    def _select_profile_for_tensor(self, handle: TRTEngineHandle,
                                   tensor_name: str,
                                   tensor_shape: tuple) -> int:
        """ Select the appropriate profile for a given tensor and shape """
        opt_profiles = self._get_opt_profiles(handle, tensor_name)

        if not opt_profiles:
            raise ValueError(
                f"No optimization profiles found for tensor '{tensor_name}'")

        # Find matching profile
        for profile_info in opt_profiles:
            if self._shape_matches_profile(tensor_shape,
                                           profile_info['min_shape'],
                                           profile_info['max_shape']):
                return profile_info['profile_idx']

        # If no exact match found, raise error with detailed information
        profile_ranges = []
        for profile_info in opt_profiles:
            profile_ranges.append(
                f"Profile {profile_info['profile_idx']}: {profile_info['min_shape']} to {profile_info['max_shape']}"
            )

        raise ValueError(
            f"No suitable optimization profile found for tensor '{tensor_name}' with shape {tensor_shape}. "
            f"Available profiles: {', '.join(profile_ranges)}")

    def switch_opt_profile(self, handle: TRTEngineHandle,
                           inputs: dict[str, torch.Tensor]):
        """ Automatically switch to the appropriate optimization profile based on input shapes """
        if not hasattr(handle, '_opt_profile_map'):
            self.build_opt_profile_map(handle)

        if not inputs:
            logger.warning("No inputs provided, using profile 0")
            if handle._curr_profile != 0:
                handle._curr_profile = 0
                with _scoped_stream() as stream:
                    handle._context.set_optimization_profile_async(0, stream)
            return

        # 1. Get shape of an input tensor from the input dictionary
        selected_tensor_name = None
        selected_tensor_shape = None

        for tensor_name, tensor in inputs.items():
            if tensor_name in handle._opt_profile_map:
                selected_tensor_name = tensor_name
                selected_tensor_shape = tuple(tensor.shape)
                break

        if selected_tensor_name is None:
            # Fallback: use any input tensor and try to find profiles
            for tensor_name, tensor in inputs.items():
                opt_profiles = self._get_opt_profiles(handle, tensor_name)
                if opt_profiles:
                    selected_tensor_name = tensor_name
                    selected_tensor_shape = tuple(tensor.shape)
                    break

        if selected_tensor_name is None:
            logger.warning(
                "Could not find any input tensor with optimization profiles, using profile 0"
            )
            if handle._curr_profile != 0:
                handle._curr_profile = 0
                with _scoped_stream() as stream:
                    handle._context.set_optimization_profile_async(0, stream)
            return

        # 2. Get optimization profiles for the selected tensor (cached for reuse)
        opt_profiles = self._get_opt_profiles(handle, selected_tensor_name)

        # 3. Select the profile that matches the shape
        selected_profile = self._select_profile_for_tensor(
            handle, selected_tensor_name, selected_tensor_shape)

        # Switch profile if needed
        if selected_profile != handle._curr_profile:
            logger.debug(
                f"Switching from profile {handle._curr_profile} to profile {selected_profile} "
                f"for tensor '{selected_tensor_name}' with shape {selected_tensor_shape}"
            )
            handle._curr_profile = selected_profile
            with _scoped_stream() as stream:
                handle._context.set_optimization_profile_async(
                    selected_profile, stream)

    def get_opt_profile_info(self, backend_impl: 'BackendBase'):
        """ Get optimization profile information for debugging """
        handle = self.get_handles()[backend_impl]
        if not hasattr(handle, '_opt_profile_map'):
            self.build_opt_profile_map(handle)

        # Convert the profile map to a more readable format for debugging
        readable_profile_map = {}
        for tensor_name, profiles in handle._opt_profile_map.items():
            readable_profile_map[tensor_name] = []
            for profile_info in profiles:
                readable_profile_map[tensor_name].append({
                    'profile_idx':
                    profile_info['profile_idx'],
                    'min_shape':
                    profile_info['min_shape'],
                    'max_shape':
                    profile_info['max_shape']
                })

        return {
            'current_profile':
            handle._curr_profile,
            'profile_map':
            readable_profile_map,
            'num_profiles':
            handle._engine.num_optimization_profiles if handle._engine else 0,
            'num_input_tensors':
            len(handle._opt_profile_map)
        }

    def load(self):
        raise NotImplementedError("load() is not implemented")

    def set_stream(self, stream: Any):
        current_stream = torch.cuda.current_stream()
        if stream is not None and stream != current_stream:
            current_stream.synchronize()
            torch.cuda.set_stream(stream)

    def unset_stream(self, stream: Any):
        current_stream = torch.cuda.current_stream()
        if stream is not None and stream != current_stream:
            stream.synchronize()
            torch.cuda.set_stream(current_stream)

    def _execute_forward(self, backend_impl: 'BackendBase',
                         inputs: dict[str, torch.Tensor]):
        # Get the engine handle for this backend
        handle = self.get_handles()[backend_impl]
        session = handle._session

        # Auto-switch to appropriate optimization profile
        self.switch_opt_profile(handle, inputs)

        # Convert inputs to the correct dtype
        trt_inputs = {
            name: tensor.to(tensor.dtype)
            for name, tensor in inputs.items()
        }

        # Get output shapes
        input_info = [
            TensorInfo(name,
                       dtype=torch_dtype_to_trt(tensor.dtype),
                       shape=tensor.shape)
            for name, tensor in trt_inputs.items()
        ]
        output_info = session.infer_shapes(input_info, session.context)

        # Create output tensors
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in output_info
        }

        # Add workspace for distributed training if needed
        if handle._world_size > 1 and not handle._disable_custom_all_reduce:
            trt_inputs["all_reduce_workspace"] = handle._workspace

        # Run inference
        ok = session.run(trt_inputs,
                         outputs,
                         handle._stream.cuda_stream,
                         context=session.context)
        assert ok, "Runtime execution failed"

        return outputs, handle


class SimpleContextMemoryManager(BaseContextMemoryManager):
    """ Note: allocate separately context memory for each engine. """

    def load(self):
        for k, v in self.get_handles().items():
            if k not in self.get_deserialized_handles():
                v.deserialize_engine()
                self.add_deserialized_handle(k, v)

    def forward(self, backend_impl: 'BackendBase', inputs: dict[str,
                                                                torch.Tensor]):
        handle = self.get_deserialized_handles().get(backend_impl)
        self.set_stream(handle.get_stream())
        try:
            outputs, _ = self._execute_forward(backend_impl, inputs)
        finally:
            self.unset_stream(handle.get_stream())
        return outputs


class SharedContextMemoryManager(BaseContextMemoryManager):
    """ Note: allocate shared context memory for all engines.
    Use the maximum size of the context memory for all engines.
    TODO: This implementation is very un-safety for progress have the torch.cuda.empty_cache() at the end,
    So we should use the OnDemandContextMemoryManager for more safety, and need to be investigated for hooking to torch.cuda.empty_cache().
    Or using the torch.cuda.memory.MemPool() to allocate the shared memory.
    """

    def __init__(self, auto_load: bool = True):
        super().__init__(auto_load)
        self._shared_memory_address = None
        self._shared_memory_size = 0

    def load(self):
        # TODO: thread-safe
        # Deserialize all engines without allocating memory
        for k, v in self.get_handles().items():
            if k not in self.get_deserialized_handles():
                v.deserialize_engine(create_without_device_memory=True,
                                     do_allocate_context_memory=False)
                self.add_deserialized_handle(k, v)

        # Find the maximum memory size required by any engine
        max_memory_size = 0
        for k, v in self.get_deserialized_handles().items():
            max_memory_size = max(max_memory_size, v.device_memory_size)

        if max_memory_size > self._shared_memory_size or self._shared_memory_address is None:
            # Allocate new shared memory
            self._shared_memory_size = max_memory_size
            logger.info(
                f"Shared context memory size: {self._shared_memory_size}")

            # Allocate the shared memory buffer
            if self._shared_memory_address is not None:
                torch.cuda.caching_allocator_delete(
                    self._shared_memory_address)
            self._shared_memory_address = torch.cuda.caching_allocator_alloc(
                self._shared_memory_size)

            # Set the shared memory for all engines
            for k, v in self.get_deserialized_handles().items():
                v.set_context_memory(self._shared_memory_address,
                                     self._shared_memory_size)
        else:
            # Reuse the existing shared memory
            for k, v in self.get_deserialized_handles().items():
                if v.device_memory_address != self._shared_memory_address:
                    v.set_context_memory(self._shared_memory_address,
                                         self._shared_memory_size)

    def forward(self, backend_impl: 'BackendBase', inputs: dict[str,
                                                                torch.Tensor]):
        handle = self.get_deserialized_handles().get(backend_impl)
        assert handle is not None, f"Engine handle for backend {backend_impl} not found"
        self.set_stream(handle.get_stream())
        try:
            outputs, handle = self._execute_forward(backend_impl, inputs)
        finally:
            self.unset_stream(handle.get_stream())
        return outputs

    def __del__(self):
        # Clean up the shared memory when the manager is destroyed
        if self._shared_memory_address is not None:
            try:
                # cudart.cudaFree(self._shared_memory_address)
                # This does not actually free the memory,
                # it just move the memory to the free list for post-processing (merge blocks memory) on torch caching allocator
                # So it's fast for almost usecases
                torch.cuda.caching_allocator_delete(
                    self._shared_memory_address)
            except:
                pass


class OnDemandContextMemoryManager(BaseContextMemoryManager):
    """ Note: allocate context memory on demand. """

    def __init__(self,
                 auto_load: bool = True,
                 clear_cache_before_forward: bool = False):
        super().__init__(auto_load)
        self._clear_cache_before_forward = clear_cache_before_forward

    def load(self):
        for k, v in self.get_handles().items():
            if k not in self.get_deserialized_handles():
                v.deserialize_engine(create_without_device_memory=True,
                                     do_allocate_context_memory=False)
                self.add_deserialized_handle(k, v)
        logger.info(
            f"Initialized {len(self.get_handles())} engines without memory allocation"
        )

    def _ensure_engine_ready(self, engine_handle: TRTEngineHandle):
        if engine_handle.device_memory_address is None:
            address = torch.cuda.caching_allocator_alloc(
                engine_handle.device_memory_size, stream=engine_handle._stream)
            logger.debug(
                f"Allocated {engine_handle.device_memory_size} bytes for engine"
            )
            engine_handle.set_context_memory(address,
                                             engine_handle.device_memory_size)

    def _cleanup_engine_memory(self, engine_handle: TRTEngineHandle):
        # Free the memory
        if engine_handle.device_memory_address is not None:
            # cudart.cudaFree(engine_handle.device_memory_address)
            torch.cuda.caching_allocator_delete(
                engine_handle.device_memory_address)
            logger.debug(f"Release memory for engine")
        # Reset the engine's memory state
        engine_handle.set_context_memory(None, 0)

    def forward(self, backend_impl: 'BackendBase', inputs: dict[str,
                                                                torch.Tensor]):
        if self._clear_cache_before_forward:
            torch.cuda.empty_cache()
        # Get the engine handle for this backend
        handle = self.get_deserialized_handles().get(backend_impl)
        assert handle is not None, f"Engine handle for backend {backend_impl} not found"
        outputs = None
        try:
            # Ensure engine has memory allocated
            self._ensure_engine_ready(handle)
            self.set_stream(handle.get_stream())
            outputs, _ = self._execute_forward(backend_impl, inputs)
        finally:
            self.unset_stream(handle.get_stream())
            self._cleanup_engine_memory(handle)
        return outputs
