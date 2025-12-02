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
from typing import Callable

import torch.nn as nn

from tensorrt_bionemo._trt.module_wrappers import EvoformerStackTRT

from ..helper import AcceleratedModules, OptimizedModuleSetterMixin


class OpenFold2AcceleratedModules(AcceleratedModules):

    def get_supported_modules(self) -> dict[str, tuple[nn.Module, Callable]]:

        def evoformer_setter(mod: nn.Module, optimized: nn.Module) -> nn.Module:
            org = mod.evoformer
            setattr(mod, "evoformer", optimized)
            return org

        return {
            "evoformer": (EvoformerStackTRT, evoformer_setter),
        }


class OpenFold2(nn.Module, OptimizedModuleSetterMixin):
    # FIXME: Implement this
    pass
