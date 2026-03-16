# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boltz1 tokenizer – wires Boltz1ContextGenerator into TRT-BNM."""

from collections import OrderedDict
from typing import Callable

from tensorrt_bionemo.pipeline.base import (ContextGeneratorSpec,
                                            TokenizerBase, TransformSpec,
                                            dict_context_merger)

from .feature_context import Boltz1ContextGenerator


class Tokenizer(TokenizerBase):
    context_generator_specs: OrderedDict[
        str, ContextGeneratorSpec] = OrderedDict({
            "primary":
            ContextGeneratorSpec(
                name="primary",
                generator=Boltz1ContextGenerator,
                required_kwargs=["parsed"],
            )
        })

    context_merger_func: Callable = dict_context_merger

    transform_specs: list[TransformSpec] = []
