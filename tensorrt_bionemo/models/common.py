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

from abc import ABC, abstractmethod
from typing import Any, Union

from tensorrt_llm.logger import logger

from tensorrt_bionemo.runtime import BackendType


class AcceleratedModules(ABC):

    def __init__(self, checkpoints: dict[str, Any],
                 backend: Union[BackendType, dict[str, BackendType]]):
        self._checkpoints = {}
        for k, v in checkpoints.items():
            if k not in self.get_supported_module_names():
                logger.warning(f"Unknown module: {k}")
            else:
                self._checkpoints[k] = v
        self._backend = {}
        if isinstance(backend, str):
            if not BackendType.is_supported(backend):
                raise ValueError(f"Unsupported backend: {backend}")
            for k, v in self._checkpoints.items():
                self._backend[k] = backend
        elif isinstance(backend, dict):
            for k, v in self._checkpoints.items():
                self._backend[k] = backend.get(k, backend)

    def get_module_backend(self, module_name: str):
        return self._backend.get(module_name, None)

    def get_module_checkpoint(self, module_name: str):
        return self._checkpoints.get(module_name, None)

    def get_module_names(self):
        return self._checkpoints.keys()

    @abstractmethod
    def get_supported_module_names(self):
        raise NotImplementedError("Subclass must implement this method")
