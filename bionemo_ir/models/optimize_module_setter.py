# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from bionemo_ir.configs import AcceleratedConfig, BackendType


class OptimizedModuleSetterMixin:
    #: Roles of the model's graph regions (:class:`GraphRegion`): role name -> region path.
    GRAPH_REGIONS: dict[str, str] = {}

    def optimize(self, accelerated_configs: dict[str, AcceleratedConfig] | None = None) -> nn.Module:
        """Enable the model's CUDA-graph regions.

        Graph regions capture the model's own loops without replacing modules; a region with a
        policy captures by default. Without ``accelerated_configs`` every region with a policy is
        enabled again. Each key names a region role (``GRAPH_REGIONS``) or path and enables that
        region, under ``acc_config.default.graph_optimization_config`` when set; ``{}`` does nothing.

        Args:
            accelerated_configs: Regions to enable, by role or path. Each value is an
                :class:`AcceleratedConfig` whose ``backend`` selects ``"torch"``.

        Returns:
            The model (self, modified in place).

        Raises:
            ValueError: If a key names no graph region, a config selects another backend, or a
                region would be left without a policy. Nothing is enabled then.
            TypeError: If a policy is neither a ``CUDAGraphOptimizationConfig`` nor a mapping of
                its fields. Nothing is enabled then.
        """
        from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig
        from bionemo_ir._torch.graph_optimization.region import GraphRegion

        if accelerated_configs is None:
            for module in self.modules():
                if isinstance(module, GraphRegion) and module.policy is not None:
                    module.enable()
            return self
        selected: list[tuple[GraphRegion, CUDAGraphOptimizationConfig | None]] = []
        unknown = []
        for name, acc_config in accelerated_configs.items():
            try:
                region = self.get_submodule(self.GRAPH_REGIONS.get(name, name))
            except AttributeError:
                region = None
            if not isinstance(region, GraphRegion):
                unknown.append(name)
                continue
            if isinstance(acc_config, dict):
                acc_config = AcceleratedConfig(**acc_config)
            if acc_config.backend != BackendType.TORCH:
                raise ValueError(f"Graph region {name!r} needs the torch backend")
            policy = getattr(acc_config.default, "graph_optimization_config", None)
            # ``BaseConfig`` types the field ``Any``, so a dict-built config carries a raw mapping.
            if isinstance(policy, dict):
                policy = CUDAGraphOptimizationConfig.model_validate(policy)
            elif policy is not None and not isinstance(policy, CUDAGraphOptimizationConfig):
                raise TypeError(
                    f"Graph region {name!r} needs a CUDAGraphOptimizationConfig policy, got {type(policy).__name__}"
                )
            if policy is None and region.policy is None:
                raise ValueError(f"Graph region {name!r} has no CUDA-graph policy")
            selected.append((region, policy))
        if unknown:
            raise ValueError(
                f"No graph region named {sorted(unknown)}; the model's region roles are {sorted(self.GRAPH_REGIONS)}"
            )
        for region, policy in selected:
            region.enable(policy)
        return self
