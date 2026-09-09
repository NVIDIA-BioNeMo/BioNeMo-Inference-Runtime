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

# Copyright 2021 DeepMind Technologies Limited
# Modified by NVIDIA Corporation and affiliates.
"""Pairing logic for multimer data pipeline."""

import collections
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg

import bionemo_ir.pipeline.models.openfold2.const as rc

# Note: these pairing thresholds and pad values are module-level constants
# rather than config fields — they match the OpenFold-2 multimer pairing
# reference and are not intended to be tuned per run.
MSA_GAP_IDX = rc.restypes_with_x_and_gap.index("-")
SEQUENCE_GAP_CUTOFF = 0.5
SEQUENCE_SIMILARITY_CUTOFF = 0.9

MSA_PAD_VALUES = {
    "msa_all_seq": MSA_GAP_IDX,
    "msa_mask_all_seq": 1,
    "deletion_matrix_all_seq": 0,
    "deletion_matrix_int_all_seq": 0,
    "msa": MSA_GAP_IDX,
    "msa_mask": 1,
    "deletion_matrix": 0,
    "deletion_matrix_int": 0,
}

MSA_FEATURES = ("msa", "msa_mask", "deletion_matrix", "deletion_matrix_int")
SEQ_FEATURES = (
    "residue_index",
    "aatype",
    "all_atom_positions",
    "all_atom_mask",
    "seq_mask",
    "between_segment_residues",
    "has_alt_locations",
    "has_hetatoms",
    "asym_id",
    "entity_id",
    "sym_id",
    "entity_mask",
    "deletion_mean",
    "prediction_atom_mask",
    "literature_positions",
    "atom_indices_to_group_indices",
    "rigid_group_default_frame",
)
TEMPLATE_FEATURES = ("template_aatype", "template_all_atom_positions", "template_all_atom_mask")
CHAIN_FEATURES = ("num_alignments", "seq_length")


def pad_features(feature: np.ndarray, feature_name: str) -> np.ndarray:
    """Add a 'padding' row at the end of the features list.

    The padding row will be selected as a 'paired' row in the case of partial
    alignment - for the chain that doesn't have paired alignment.

    Args:
        feature: The feature to be padded.
        feature_name: The name of the feature to be padded.

    Returns:
        The feature with an additional padding row.
    """
    assert feature.dtype != np.dtype(np.bytes_)

    if feature_name in ("msa_all_seq", "msa_mask_all_seq", "deletion_matrix_all_seq", "deletion_matrix_int_all_seq"):
        num_res = feature.shape[1]
        padding = MSA_PAD_VALUES[feature_name] * np.ones([1, num_res], feature.dtype)
    elif feature_name == "msa_species_identifiers_all_seq":
        padding = [b""]
    else:
        return feature

    feats_padded = np.concatenate([feature, padding], axis=0)
    return feats_padded


def block_diag(*arrs: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Like scipy.linalg.block_diag but with an optional padding value."""
    ones_arrs = [np.ones_like(x) for x in arrs]
    off_diag_mask = 1.0 - scipy.linalg.block_diag(*ones_arrs)
    diag = scipy.linalg.block_diag(*arrs)
    diag += (off_diag_mask * pad_value).astype(diag.dtype)
    return diag


def _correct_post_merged_feats(
    np_example: Mapping[str, np.ndarray], np_chains_list: Sequence[Mapping[str, np.ndarray]], pair_msa_sequences: bool
) -> Mapping[str, np.ndarray]:
    """Adds features that need to be computed/recomputed post merging."""
    np_example["seq_length"] = np.asarray(np_example["aatype"].shape[0], dtype=np.int32)
    np_example["num_alignments"] = np.asarray(np_example["msa"].shape[0], dtype=np.int32)

    if not pair_msa_sequences:
        # Generate a bias that is 1 for the first row of every block in the
        # block diagonal MSA - i.e. make sure the cluster stack always includes
        # the query sequences for each chain (since the first row is the query
        # sequence).
        cluster_bias_masks = []
        for chain in np_chains_list:
            mask = np.zeros(chain["msa"].shape[0])
            mask[0] = 1
            cluster_bias_masks.append(mask)

        np_example["cluster_bias_mask"] = np.concatenate(cluster_bias_masks)

        # Initialize Bert mask with masked out off diagonals.
        msa_masks = [np.ones(x["msa"].shape, dtype=np.float32) for x in np_chains_list]
        np_example["bert_mask"] = block_diag(*msa_masks, pad_value=0)
    else:
        np_example["cluster_bias_mask"] = np.zeros(np_example["msa"].shape[0])
        np_example["cluster_bias_mask"][0] = 1

        # Initialize Bert mask with masked out off diagonals.
        msa_masks = [np.ones(x["msa"].shape, dtype=np.float32) for x in np_chains_list]
        msa_masks_all_seq = [np.ones(x["msa_all_seq"].shape, dtype=np.float32) for x in np_chains_list]

        msa_mask_block_diag = block_diag(*msa_masks, pad_value=0)
        msa_mask_all_seq = np.concatenate(msa_masks_all_seq, axis=1)
        np_example["bert_mask"] = np.concatenate([msa_mask_all_seq, msa_mask_block_diag], axis=0)

    return np_example


def _pad_templates(
    chains: Sequence[Mapping[str, np.ndarray]], max_templates: int
) -> Sequence[Mapping[str, np.ndarray]]:
    """For each chain pad the number of templates to a fixed size.

    Args:
        chains: A list of protein chains.
        max_templates: Each chain will be padded to have this many templates.

    Returns:
        The list of chains, updated to have template features padded to
        max_templates.
    """
    for chain in chains:
        for k, v in chain.items():
            if k in TEMPLATE_FEATURES:
                padding = np.zeros(len(v.shape), dtype=int)
                padding[0] = max_templates - v.shape[0]
                padding = [(0, p) for p in padding]
                chain[k] = np.pad(v, padding, mode="constant")
    return chains


def _merge_features_from_multiple_chains(
    chains: Sequence[Mapping[str, np.ndarray]], pair_msa_sequences: bool
) -> Mapping[str, np.ndarray]:
    """Merge features from multiple chains.

    Args:
        chains: A list of feature dictionaries that we want to merge.
        pair_msa_sequences: Whether to concatenate MSA features along the
            num_res dimension (if True), or to block diagonalize them (if False).

    Returns:
        A feature dictionary for the merged example.
    """
    merged_example = {}
    for feature_name in chains[0]:
        feats = [x[feature_name] for x in chains]
        feature_name_split = feature_name.split("_all_seq")[0]
        if feature_name_split in MSA_FEATURES:
            if pair_msa_sequences or "_all_seq" in feature_name:
                merged_example[feature_name] = np.concatenate(feats, axis=1)
            else:
                merged_example[feature_name] = block_diag(*feats, pad_value=MSA_PAD_VALUES[feature_name])
        elif feature_name_split in SEQ_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=0)
        elif feature_name_split in TEMPLATE_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=1)
        elif feature_name_split in CHAIN_FEATURES:
            merged_example[feature_name] = np.sum(feats).astype(np.int32)
        else:
            merged_example[feature_name] = feats[0]
    return merged_example


def _merge_homomers_dense_msa(chains: Iterable[Mapping[str, np.ndarray]]) -> Sequence[Mapping[str, np.ndarray]]:
    """Merge all identical chains, making the resulting MSA dense.

    Args:
        chains: An iterable of features for each chain.

    Returns:
        A list of feature dictionaries. All features with the same entity_id
        will be merged - MSA features will be concatenated along the num_res
        dimension - making them dense.
    """
    entity_chains = collections.defaultdict(list)
    for chain in chains:
        entity_id = chain["entity_id"][0]
        entity_chains[entity_id].append(chain)

    grouped_chains = []
    for entity_id in sorted(entity_chains):
        chains = entity_chains[entity_id]
        grouped_chains.append(chains)

    chains = [_merge_features_from_multiple_chains(chains, pair_msa_sequences=True) for chains in grouped_chains]
    return chains


def _concatenate_paired_and_unpaired_features(example: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Merges paired and block-diagonalised features."""
    features = MSA_FEATURES
    for feature_name in features:
        if feature_name in example:
            feat = example[feature_name]
            feat_all_seq = example[feature_name + "_all_seq"]
            merged_feat = np.concatenate([feat_all_seq, feat], axis=0)
            example[feature_name] = merged_feat
    example["num_alignments"] = np.array(example["msa"].shape[0], dtype=np.int32)
    return example


def merge_chain_features(
    np_chains_list: list[Mapping[str, np.ndarray]], pair_msa_sequences: bool, max_templates: int
) -> Mapping[str, np.ndarray]:
    """Merges features for multiple chains to single FeatureDict.

    Args:
        np_chains_list: List of FeatureDicts for each chain.
        pair_msa_sequences: Whether to merge paired MSAs.
        max_templates: The maximum number of templates to include.

    Returns:
        Single FeatureDict for entire complex.
    """
    np_chains_list = _pad_templates(np_chains_list, max_templates=max_templates)
    np_chains_list = _merge_homomers_dense_msa(np_chains_list)

    # Unpaired MSA features will be always block-diagonalised; paired MSA
    # features will be concatenated.
    np_example = _merge_features_from_multiple_chains(np_chains_list, pair_msa_sequences=False)
    if pair_msa_sequences:
        np_example = _concatenate_paired_and_unpaired_features(np_example)

    np_example = _correct_post_merged_feats(
        np_example=np_example, np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences
    )

    return np_example
