# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boltz2 tokenizer transforms.

Boltz2 does not require any post-context-generation transforms; all feature
computation is handled inside the ContextGenerator (which wraps the OSS
tokenizer + featurizer).  This module is intentionally empty and exists only
for architectural completeness.
"""
