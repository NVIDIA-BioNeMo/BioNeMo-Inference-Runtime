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

from collections import OrderedDict
from typing import Callable

from tensorrt_bionemo.pipeline.base import (ContextGeneratorSpec,
                                            TokenizerBase, TransformSpec,
                                            dict_context_merger)

from .context import (MSAContextGenerator, PrimaryContextGenerator,
                      TemplateContextGenerator)
from .transforms import (CastTo64BitInts, CorrectMsaRestypes,
                         FixTemplatesAatype, RandomlyReplaceMsaWithUnknown,
                         SqueezeFeatures)


class Tokenizer(TokenizerBase):
    context_generator_specs: OrderedDict[
        str, ContextGeneratorSpec] = OrderedDict({
            'primary':
            ContextGeneratorSpec(name='primary',
                                 generator=PrimaryContextGenerator,
                                 required_kwargs=['parsed']),
            'msa':
            ContextGeneratorSpec(name='msa',
                                 generator=MSAContextGenerator,
                                 required_kwargs=['parsed']),
            'template':
            ContextGeneratorSpec(name='template',
                                 generator=TemplateContextGenerator,
                                 required_kwargs=['parsed']),
        })

    context_merger_func: Callable = dict_context_merger

    transform_specs: list[TransformSpec] = [
        TransformSpec(name='cast_to_64_bit_ints', transform=CastTo64BitInts),
        TransformSpec(name='correct_msa_restypes',
                      transform=CorrectMsaRestypes),
        TransformSpec(name='squeeze_features', transform=SqueezeFeatures),
        TransformSpec(name='randomly_replace_msa_with_unknown',
                      transform=RandomlyReplaceMsaWithUnknown,
                      kwargs={'replace_proportion': 0.0}),
        TransformSpec(name='fix_templates_aatype',
                      transform=FixTemplatesAatype),
    ]
