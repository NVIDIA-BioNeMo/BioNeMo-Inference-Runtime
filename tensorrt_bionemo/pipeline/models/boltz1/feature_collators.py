# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boltz1 feature collator – identical to Boltz2."""

from tensorrt_bionemo.pipeline.models.boltz2.feature_collators import \
    Boltz2FinalFeatureCollator as Boltz1FinalFeatureCollator

__all__ = ["Boltz1FinalFeatureCollator"]
