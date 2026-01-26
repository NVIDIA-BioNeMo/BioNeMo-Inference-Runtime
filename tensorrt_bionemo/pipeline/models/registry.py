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
from tensorrt_bionemo.logger import logger
from tensorrt_bionemo.pipeline.base import (FeatureFactoryBase,
                                            PostProcessorBase, TokenizerBase)

TOKENIZER_REGISTRY = {}
FEATURE_FACTORY_REGISTRY = {}
POSTPROCESSOR_REGISTRY = {}


def get_postprocessor(model_name: str) -> PostProcessorBase:
    if model_name not in POSTPROCESSOR_REGISTRY:
        raise ValueError(f"Postprocessor for model {model_name} not found")
    return POSTPROCESSOR_REGISTRY[model_name]


def register_postprocessor(model_name: str, postprocessor: PostProcessorBase):
    if model_name in POSTPROCESSOR_REGISTRY:
        logger.warning(
            f"Postprocessor for model {model_name} already registered, overriding with new postprocessor"
        )
    else:
        logger.info(f"Registering postprocessor for model {model_name}")
    POSTPROCESSOR_REGISTRY[model_name] = postprocessor


def get_feature_factory(model_name: str) -> FeatureFactoryBase:
    if model_name not in FEATURE_FACTORY_REGISTRY:
        raise ValueError(f"Feature factory for model {model_name} not found")
    return FEATURE_FACTORY_REGISTRY[model_name]


def register_feature_factory(model_name: str,
                             feature_factory: FeatureFactoryBase):
    if model_name in FEATURE_FACTORY_REGISTRY:
        logger.warning(
            f"Feature factory for model {model_name} already registered, overriding with new feature factory"
        )
    else:
        logger.info(f"Registering feature factory for model {model_name}")
    FEATURE_FACTORY_REGISTRY[model_name] = feature_factory


def get_tokenizer(model_name: str) -> TokenizerBase:
    if model_name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Tokenizer for model {model_name} not found")
    return TOKENIZER_REGISTRY[model_name]


def register_tokenizer(model_name: str, tokenizer: TokenizerBase):
    if model_name in TOKENIZER_REGISTRY:
        logger.warning(
            f"Tokenizer for model {model_name} already registered, overriding with new tokenizer"
        )
    else:
        logger.info(f"Registering tokenizer for model {model_name}")
    TOKENIZER_REGISTRY[model_name] = tokenizer


def register_default_tokenizers():
    pass


def register_default_feature_factories():
    pass


def register_default_postprocessors():
    pass


register_default_tokenizers()
register_default_feature_factories()
register_default_postprocessors()
