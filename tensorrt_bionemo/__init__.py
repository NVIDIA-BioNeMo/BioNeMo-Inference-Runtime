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

import logging
import os
from pathlib import Path

_log_level = os.environ.get("TENSORRT_BIONEMO_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

CACHE_DIR = Path(
    os.getenv("TENSORRT_BIONEMO_CACHE",
              str(Path.home() / ".cache" / "tensorrt_bionemo")))

from ._torch import _load_cuequivariance_lib, _load_kernels_lib
from ._trt.plugin import _load_plugin_lib
from .registry import register_all_factories
from .version import __version__

_inited = False


def _init() -> None:
    global _inited
    if _inited:
        return
    _inited = True
    # load CuEquivariance Ops library
    _load_cuequivariance_lib()
    # load TensorRT-BioNemo Kernels library
    _load_kernels_lib()
    # load Tensorrt plugins library
    _load_plugin_lib()
    register_all_factories()


_init()

import tensorrt_bionemo._trt.layers as layers

__all__ = ["layers", "__version__", "CACHE_DIR"]
