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

"""Sequence-alignment primitives shared across template featurizers.

Model-agnostic helpers that turn a pairwise (kalign) alignment into a 1-based
residue index map and score it. Vectorized port of OpenFold ``calculate_ids_hit``
/ ``compute_sequence_identity_and_coverage``:
https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/data/templates.py
"""

from __future__ import annotations

import numpy as np

# Alignment gap characters (kalign emits '-'; '.' guards insertion states).
GAP_CHARS = ("-", ".")


def calculate_ids_hit(
    query_aligned: np.ndarray,
    template_aligned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 1-based residue indices for two equal-length aligned sequences.

    Gaps are encoded as ``-1`` and columns where both sequences are gaps are
    dropped. Inputs are ``<U1`` character arrays of equal length.
    """
    query_is_residue = ~np.isin(query_aligned, GAP_CHARS)
    template_is_residue = ~np.isin(template_aligned, GAP_CHARS)
    columns_to_keep = query_is_residue | template_is_residue
    query_ids = np.where(query_is_residue, np.cumsum(query_is_residue), -1)
    template_ids = np.where(template_is_residue, np.cumsum(template_is_residue), -1)
    return query_ids[columns_to_keep], template_ids[columns_to_keep]


def seq_identity_and_coverage(
    query_aligned: np.ndarray,
    template_aligned: np.ndarray,
    query_seq: str,
) -> tuple[float, float]:
    """Sequence identity (over aligned query residues) and query coverage."""
    query_is_residue = ~np.isin(query_aligned, GAP_CHARS)
    query_residue_count = int(query_is_residue.sum())
    if query_residue_count == 0:
        return 0.0, 0.0
    identical = int((template_aligned == query_aligned)[query_is_residue].sum())
    mutually_aligned = int((query_is_residue & ~np.isin(template_aligned, GAP_CHARS)).sum())
    return identical / query_residue_count, mutually_aligned / max(len(query_seq), 1)
