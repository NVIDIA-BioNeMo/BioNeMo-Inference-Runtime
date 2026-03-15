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
import torch.nn as nn

from tensorrt_bionemo._trt.module_wrappers import (PairformerTRT,
                                                   TokenTransformerTRT)

from ..helper import (AcceleratedConfig, ModuleRegistry, ModuleSpec,
                      OptimizedModuleSetterMixin)


class OpenFold3ModuleRegistry(ModuleRegistry):

    def get_accelerated_modules(self) -> dict[str, ModuleSpec]:
        return {
            "pairformer":
            ModuleSpec(
                getter=lambda mod: mod.pairformer_stack,
                setter=lambda mod, opt: setattr(mod, "pairformer_stack", opt),
                trt_cls=PairformerTRT,
                compiled_cls=None,
            ),
            "token_transformer":
            ModuleSpec(
                getter=lambda mod:
                (mod.sample_diffusion.diffusion_module.diffusion_transformer),
                setter=lambda mod, opt: setattr(
                    mod.sample_diffusion.diffusion_module,
                    "diffusion_transformer", opt),
                trt_cls=TokenTransformerTRT,
                compiled_cls=None,
            ),
        }


class OpenFold3(nn.Module, OptimizedModuleSetterMixin):

    def get_optimized_modules(
        self, accelerated_configs: dict[str, AcceleratedConfig]
    ) -> OpenFold3ModuleRegistry:
        return OpenFold3ModuleRegistry(accelerated_configs)
