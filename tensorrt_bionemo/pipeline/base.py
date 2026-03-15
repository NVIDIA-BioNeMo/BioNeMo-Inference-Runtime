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

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import torch
from pydantic import BaseModel, Field

from tensorrt_bionemo.configs.base import BaseConfig

# Feature preprocessing pipeline stages:
# 1. Tokenizer:
#   - Context Generator: Generates context tensors from input data.
#   - Context Merger: Merges context tensors from multiple sources.
#   - Context Transform: Normalizes and transforms context tensors.
#     Ideally, it shouldn't add new tensors but optional by user intention.
# 2. Feature Generator:
#   - Feature Generator: Generates feature tensors based on context tensors.
#     The output of this stage is a dictionary of tensors (context tensors + generated tensors).
#     The reason why we need to separate the stage because it can run in parallel.
#   - Context and Feature Merger: Merges context tensors and generated feature tensors from multiple generators
#   - Feature Collator: Collates feature tensors to produce the final feature set.
#     Conceptually, this step can add new features, remove features, or modify the features.


class FeatureGeneratorBase(ABC):
    """
    Abstract base class for transform functions.
    """

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 metadata: Optional[dict[str, Any]] = None,
                 **kwargs: Any):
        self.config = config
        self._name = kwargs.get("name", self.__class__.__name__)
        self.metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Apply the transform to the batch.
        """
        return batch

    def is_enabled(self) -> bool:
        """
        Check if the transform is enabled.
        """
        return True


class FeatureCollatorBase(FeatureGeneratorBase):
    pass


class FeatureGeneratorSpec(BaseModel):
    name: Optional[str] = Field(
        description="The name of the feature generator/collator.",
        default="generator")
    functor: Type[FeatureGeneratorBase] = Field(
        description="The feature generator/collator function.")
    kwargs: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="The kwargs for the feature generator/collator.")


class FeatureCollatorSpec(FeatureGeneratorSpec):
    pass


class FeatureFactoryBase(BaseModel):
    """ Workflow: pre_init -> feature_generator -> features_merger -> feature_collator
    pre_init: Setup some params for feature generator and collator.
    feature_generator: Generate the feature tensors.
    features_merger: Merge the feature tensors.
    feature_collator: Collate the feature tensors.
    """
    pre_init: Callable = Field(
        default_factory=lambda: None,
        description=
        "The function to call before the feature generator is called.")
    feature_generator_specs: list[FeatureGeneratorSpec] = Field(
        description="The dictionary of feature generator specs.")
    features_merger_func: Callable = Field(
        description="The function to merge the feature tensors.")
    feature_collator_specs: list[FeatureCollatorSpec] = Field(
        description="The dictionary of feature collator specs.")


class ContextGeneratorBase(ABC):
    """
    Abstract base class for structure context.
    """

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 metadata: Optional[dict[str, Any]] = None):
        self.config = config
        self.metadata = metadata
        self._required_kwargs = []

    @abstractmethod
    def __call__(self) -> dict[str, torch.Tensor]:
        """
        Generate a structure context from a InputRequest.
        """
        return {}

    @property
    def required_kwargs(self) -> list[str]:
        return self._required_kwargs

    @required_kwargs.setter
    def required_kwargs(self, required_kwargs: list[str]):
        self._required_kwargs = required_kwargs


class ContextGeneratorSpec(BaseModel):
    """ Context generator spec is a specification for a context generator.
    Normally, the contexts include:
      - Polymers context: the context of the polymers.
      - MSA context: the context of the MSA for the polymers.
      - Template context: the context of the template for the polymers.
    """
    name: str = Field(description="The name of the context generator.")
    generator: Type[ContextGeneratorBase] = Field(
        description="The generator function for the context generator.")
    required_kwargs: list[str] = Field(
        description="The required kwargs for the context generator.")


def dict_context_merger(
    contexts: Union[list[dict[str, torch.Tensor]],
                    dict[str, dict[str, torch.Tensor]]]
) -> dict[str, torch.Tensor]:
    """
    Merge a dictionary of context tensors into a single dictionary of context tensors.
    WARNING: This function is not key-conflict safe.
    """
    ret = {}
    if isinstance(contexts, list):
        values = contexts
    elif isinstance(contexts, dict):
        values = contexts.values()
    else:
        raise ValueError(f"Invalid type of contexts: {type(contexts)}")
    for context in values:
        ret.update(context)

    return ret


def default_context_and_feature_merger(
        contexts: dict[str, torch.Tensor],
        features: dict[str, dict[str,
                                 torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Merge the context tensors and generated feature tensors from multiple generators.
    """
    merged = contexts
    merged.update(dict_context_merger(features))
    return merged


class TransformBase(ABC):
    """
    Abstract base class for transform functions.
    """

    def __init__(self, config: Optional[BaseConfig] = None, **kwargs: Any):
        self.config = config

    @abstractmethod
    def __call__(self, batch: dict[str,
                                   torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply the transform to the batch.
        """
        return batch

    def is_enabled(self) -> bool:
        """
        Check if the transform is enabled.
        """
        return True


class TransformSpec(BaseModel):
    name: Optional[str] = Field(description="The name of the transform.",
                                default="transform")
    transform: Type[TransformBase] = Field(
        description="The transform function to apply to the batch.")
    kwargs: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="The kwargs for the transform.")


class TokenizerBase(BaseModel):
    """Workflow: context_generator -> context_merger -> context_transform """
    context_generator_specs: OrderedDict[str, ContextGeneratorSpec] = Field(
        description="The dictionary of context generator specs.")
    context_merger_func: Callable = Field(
        description="The function to merge the context tensors.")
    transform_specs: list[TransformSpec] = Field(
        description=
        "The list of transform specs to apply to the final context tensors.")


class PostProcessorBase:
    """
    Abstract base class for postprocessor functions.
    """

    def __init__(self, config: Optional[BaseModel] = None, **kwargs: Any):
        self.config = config

    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor],
                 output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply the postprocessor to the batch.
        """
        return output


def numpy_to_dict(data: Any) -> Any:
    """Recursively convert numpy arrays and structured data to Python dicts/lists."""
    if isinstance(data, np.ndarray):
        # 0-D arrays: convert to a Python scalar first
        if data.ndim == 0:
            return numpy_to_dict(data.item())
        # N-D arrays: convert to nested lists and recurse
        return [numpy_to_dict(item) for item in data.tolist()]
    elif isinstance(data, dict):
        # Recursively process dictionary values
        return {key: numpy_to_dict(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        # Recursively process list/tuple items
        return [numpy_to_dict(item) for item in data]
    else:
        # Return primitive types as-is
        return data
