# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field


class _StageConfigBase(BaseModel):
    enabled: bool = Field(default=True,
                          description="Whether this stage is enabled.")
    # Optional overrides; processor-level defaults still apply
    batch_size: Optional[int] = Field(default=None,
                                      description="Rows per batch.")
    compute: Optional[Union[int, Tuple[int, int]]] = Field(
        default=None, description="Actor pool size or range for this stage.")
    runtime_env: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional runtime env for this stage.")
    num_cpus: Optional[float] = Field(
        default=None,
        description=
        "Number of CPUs to reserve for each map worker in this stage.",
    )
    memory: Optional[float] = Field(
        default=None,
        description=
        "Heap memory in bytes to reserve for each map worker in this stage.",
    )
    compute_by_rows: bool = Field(
        default=True,
        description="Convert to rows mode and compute.",
    )
    drop_keys: Optional[List[str]] = Field(
        default=None,
        description="The keys to drop from the output.",
    )


T = TypeVar("T", bound="_StageConfigBase")


class ParserStageConfig(_StageConfigBase):
    pass


class TokenizerStageConfig(_StageConfigBase):
    pass


class FeatureGeneratorStageConfig(_StageConfigBase):
    init_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="The context to initialize the feature generator with.",
    )


class WriterStageConfig(_StageConfigBase):
    output_path: Optional[str] = Field(
        default=None,
        description="The path to write the output to.",
    )
    format: Optional[str] = Field(
        default=None,
        description="The format to write the output in.",
    )


def resolve_stage_config(
    stage_cfg_value: Union[bool, Dict[str, Any], _StageConfigBase],
    stage_config_cls: Type[T],
    processor_defaults: Optional[Dict[str, Any]] = None,
) -> T:
    """Resolve a stage config value (bool | dict | StageConfig) into a typed StageConfig.

    Args:
        stage_cfg_value: The stage config value (bool, dict, or typed StageConfig).
        stage_config_cls: The StageConfig class to instantiate.
        processor_defaults: Optional dict of processor-level defaults to merge in.
            Expected keys: 'batch_size', 'compute', 'runtime_env', 'model_source'.

    Returns:
        Resolved StageConfig instance with defaults merged.
    """
    processor_defaults = processor_defaults or {}

    # If already a typed config, create a copy to avoid mutating the input
    if isinstance(stage_cfg_value, stage_config_cls):
        resolved = stage_config_cls.model_validate(
            stage_cfg_value.model_dump())
    # If bool, create minimal config with enabled flag
    elif isinstance(stage_cfg_value, bool):
        resolved = stage_config_cls(enabled=stage_cfg_value)
    # If dict, parse it into the config class
    elif isinstance(stage_cfg_value, dict):
        resolved = stage_config_cls(**stage_cfg_value)
    else:
        raise TypeError(
            f"Unsupported type for stage config: {type(stage_cfg_value).__name__}. "
            f"Expected bool, dict, or {stage_config_cls.__name__} instance. "
            f"Got: {stage_cfg_value}")

    # Merge processor defaults for fields not explicitly set
    default_fields = ["batch_size", "compute", "runtime_env", "model_source"]
    for field_name in default_fields:
        # Skip if field doesn't exist on this config class (e.g., model_source only on some stages)
        if not hasattr(resolved, field_name):
            continue
        if (getattr(resolved, field_name, None) is None
                and field_name in processor_defaults):
            setattr(resolved, field_name, processor_defaults[field_name])

    return resolved
