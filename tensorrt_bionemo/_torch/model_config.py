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

from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import transformers

from tensorrt_bionemo.mapping import Mapping

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)
    skip_create_weights: bool = False
    triangle_attn_node_chunk_size: int = 0
    attn_backend: str = 'VANILLA'

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str,
                        trust_remote_code=False,
                        **kwargs):
        pretrained_config = transformers.AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=trust_remote_code,
        )

        return cls(pretrained_config=pretrained_config, **kwargs)
