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


import itertools
from functools import reduce
from operator import add
from typing import Any, Optional

import numpy as np
import torch

from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import FeatureCollatorBase

from .common import make_one_hot, shaped_categorical, unsorted_segment_sum

MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]


class SampleMsa(FeatureCollatorBase):

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 keep_extra: bool = True):
        super().__init__(config)
        self.keep_extra = keep_extra

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        seed = None
        if not self.config.resample_msa_in_recycling:
            seed = context.get("ensemble_seed", None)
        max_seq = self.config.max_msa_clusters
        num_seq = features["msa"].shape[0]

        g = None
        if seed is not None:
            g = torch.Generator(device=features["msa"].device)
            g.manual_seed(seed)

        shuffled = torch.randperm(num_seq - 1, generator=g) + 1
        index_order = torch.cat(
            (torch.tensor([0], device=shuffled.device), shuffled), dim=0)
        num_sel = min(max_seq, num_seq)
        sel_seq, not_sel_seq = torch.split(index_order,
                                           [num_sel, num_seq - num_sel])

        for k in MSA_FEATURE_NAMES:
            if k in list(features.keys()):
                if self.keep_extra:
                    features["extra_" + k] = torch.index_select(
                        features[k], 0, not_sel_seq)
                features[k] = torch.index_select(features[k], 0, sel_seq)

        return features


class MakeMaskedMsa(FeatureCollatorBase):

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 profile_prob: Optional[float] = 0.1,
                 same_prob: Optional[float] = 0.1,
                 uniform_prob: Optional[float] = 0.1,
                 masked_msa_replace_fraction: Optional[float] = 0.15):
        super().__init__(config)
        self.profile_prob = profile_prob
        self.same_prob = same_prob
        self.uniform_prob = uniform_prob
        self.masked_msa_replace_fraction = masked_msa_replace_fraction

    def is_enabled(self) -> bool:
        if self.profile_prob is None or \
           self.same_prob is None or \
           self.uniform_prob is None or \
           self.masked_msa_replace_fraction is None:
            return False
        return True

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Create data for BERT on raw MSA."""
        seed = None
        if not self.config.resample_msa_in_recycling:
            seed = context.get("ensemble_seed", None)
            seed = (seed + 1) if seed else None

        device = features["msa"].device

        # Add a random amino acid uniformly.
        random_aa = torch.tensor([0.05] * 20 + [0.0, 0.0],
                                 dtype=torch.float32,
                                 device=device)

        categorical_probs = (
            self.uniform_prob * random_aa +
            self.profile_prob * features["hhblits_profile"] +
            self.same_prob * make_one_hot(features["msa"], 22))

        # Put all remaining probability on [MASK] which is a new column
        pad_shapes = list(
            reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
        pad_shapes[1] = 1
        mask_prob = (1.0 - self.profile_prob - self.same_prob -
                     self.uniform_prob)
        assert mask_prob >= 0.0

        categorical_probs = torch.nn.functional.pad(
            categorical_probs,
            pad_shapes,
            value=mask_prob,
        )

        sh = features["msa"].shape

        g = None
        if seed is not None:
            g = torch.Generator(device=features["msa"].device)
            g.manual_seed(seed)

        sample = torch.rand(sh, device=device, generator=g)
        mask_position = sample < self.masked_msa_replace_fraction
        bert_msa = shaped_categorical(categorical_probs)
        bert_msa = torch.where(mask_position, bert_msa, features["msa"])

        # Mix real and masked MSA
        features["bert_mask"] = mask_position.to(torch.float32)
        features["true_msa"] = features["msa"]
        features["msa"] = bert_msa

        return features


class NearestNeighborClusters(FeatureCollatorBase):

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 gap_agreement_weight: float = 0.0):
        super().__init__(config)
        self.gap_agreement_weight = gap_agreement_weight

    def is_enabled(self) -> bool:
        return self.config.msa_cluster_features

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        weights = torch.cat(
            [
                torch.ones(21, device=features["msa"].device),
                self.gap_agreement_weight *
                torch.ones(1, device=features["msa"].device),
                torch.zeros(1, device=features["msa"].device)
            ],
            0,
        )

        # Make agreement score as weighted Hamming distance
        msa_one_hot = make_one_hot(features["msa"], 23)
        sample_one_hot = features["msa_mask"][:, :, None] * msa_one_hot
        extra_msa_one_hot = make_one_hot(features["extra_msa"], 23)
        extra_one_hot = features["extra_msa_mask"][:, :,
                                                   None] * extra_msa_one_hot

        num_seq, num_res, _ = sample_one_hot.shape
        extra_num_seq, _, _ = extra_one_hot.shape

        # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
        # in an optimized fashion to avoid possible memory or computation blowup.
        agreement = torch.matmul(
            torch.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
            torch.reshape(sample_one_hot * weights,
                          [num_seq, num_res * 23]).transpose(0, 1),
        )

        # Assign each sequence in the extra sequences to the closest MSA sample
        features["extra_cluster_assignment"] = torch.argmax(
            agreement, dim=1).to(torch.int64)

        return features


class SummarizeClusters(FeatureCollatorBase):

    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)

    def is_enabled(self) -> bool:
        return self.config.msa_cluster_features

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Produce profile and deletion_matrix_mean within each cluster."""
        num_seq = features["msa"].shape[0]

        def csum(x):
            return unsorted_segment_sum(x,
                                        features["extra_cluster_assignment"],
                                        num_seq)

        mask = features["extra_msa_mask"]
        mask_counts = 1e-6 + features["msa_mask"] + csum(
            mask)  # Include center

        msa_sum = csum(mask[:, :, None] *
                       make_one_hot(features["extra_msa"], 23))
        msa_sum += make_one_hot(features["msa"], 23)  # Original sequence
        features["cluster_profile"] = msa_sum / mask_counts[:, :, None]
        del msa_sum

        del_sum = csum(mask * features["extra_deletion_matrix"])
        del_sum += features["deletion_matrix"]  # Original sequence
        features["cluster_deletion_mean"] = del_sum / mask_counts
        del del_sum

        return features


class CropExtraMsa(FeatureCollatorBase):

    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)

    def is_enabled(self) -> bool:
        max_extra = self.config.max_extra_msa
        return max_extra is not None and max_extra > 0

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        num_seq = features["extra_msa"].shape[0]
        num_sel = min(self.config.max_extra_msa, num_seq)
        select_indices = torch.randperm(num_seq)[:num_sel]
        for k in MSA_FEATURE_NAMES:
            if "extra_" + k in features:
                features["extra_" + k] = torch.index_select(
                    features["extra_" + k], 0, select_indices)

        return features


class DeleteExtraMsa(FeatureCollatorBase):

    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)

    def is_enabled(self) -> bool:
        return self.config.max_extra_msa is None

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        for k in MSA_FEATURE_NAMES:
            if "extra_" + k in features:
                del features["extra_" + k]
        return features


class MakeMsaFeat(FeatureCollatorBase):

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Create and concatenate MSA features."""
        # Whether there is a domain break. Always zero for chains, but keeping for
        # compatibility with domain datasets.
        has_break = torch.clip(
            features["between_segment_residues"].to(torch.float32), 0, 1)
        aatype_1hot = make_one_hot(features["aatype"], 21)

        target_feat = [
            torch.unsqueeze(has_break, dim=-1),
            aatype_1hot,  # Everyone gets the original sequence.
        ]
        msa_1hot = make_one_hot(features["msa"], 23)
        has_deletion = torch.clip(features["deletion_matrix"], 0.0, 1.0)
        deletion_value = torch.atan(
            features["deletion_matrix"] / 3.0) * (2.0 / np.pi)

        msa_feat = [
            msa_1hot,
            torch.unsqueeze(has_deletion, dim=-1),
            torch.unsqueeze(deletion_value, dim=-1),
        ]

        if "cluster_profile" in features:
            deletion_mean_value = torch.atan(
                features["cluster_deletion_mean"] / 3.0) * (2.0 / np.pi)
            msa_feat.extend([
                features["cluster_profile"],
                torch.unsqueeze(deletion_mean_value, dim=-1),
            ])

        if "extra_deletion_matrix" in features:
            features["extra_has_deletion"] = torch.clip(
                features["extra_deletion_matrix"], 0.0, 1.0)
            features["extra_deletion_value"] = torch.atan(
                features["extra_deletion_matrix"] / 3.0) * (2.0 / np.pi)

        features["msa_feat"] = torch.cat(msa_feat, dim=-1)
        features["target_feat"] = torch.cat(target_feat, dim=-1)
        return features


class SelectFeat(FeatureCollatorBase):

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 include_feats: list[str] = None,
                 exclude_feats: list[str] = None):
        super().__init__(config)
        self.include_feats = include_feats
        self.exclude_feats = exclude_feats

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        remove_keys = set()

        if self.include_feats is not None:
            for key in features.keys():
                if key not in self.include_feats:
                    remove_keys.add(key)
        if self.exclude_feats is not None:
            for key in features.keys():
                if key in self.exclude_feats:
                    remove_keys.add(key)
        for key in remove_keys:
            del features[key]
        return features


class RandomCropToSize(FeatureCollatorBase):
    """ Do only the templates cropping. Due to we consider only the inference phase """

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 subsample_templates: bool = False):
        super().__init__(config)
        self.subsample_templates = subsample_templates
        self.max_templates = self.config.max_templates

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        seed = context.get("ensemble_seed", None)
        if seed is not None:
            seed = seed + 1


        seq_length = features["seq_length"]
        g = None
        if seed is not None:
            g = torch.Generator(device=seq_length.device)
            g.manual_seed(seed)

        if "template_mask" in features:
            num_templates = features["template_mask"].shape[-1]
        else:
            num_templates = 0

        subsample_templates = self.subsample_templates and num_templates

        def _randint(lower, upper):
            return int(
                torch.randint(
                    lower,
                    upper + 1,
                    (1, ),
                    device=seq_length.device,
                    generator=g,
                )[0])

        if subsample_templates:
            templates_crop_start = _randint(0, num_templates)
            templates_select_indices = torch.randperm(num_templates,
                                                      device=seq_length.device,
                                                      generator=g)
        else:
            templates_crop_start = 0

        num_templates_crop_size = min(num_templates - templates_crop_start,
                                      self.max_templates)

        for k, v in features.items():
            if "template" not in k:
                continue
            # randomly permute the templates before cropping them.
            if subsample_templates:
                v = v[templates_select_indices]

            crop_size = num_templates_crop_size
            crop_start = templates_crop_start
            # clone the tensor, because the original tensor may be unmutable
            features[k] = v[crop_start:crop_start + crop_size].clone()

        return features


class MakeFixedSize(FeatureCollatorBase):
    """ Do only the templates padding and msa padding, ignore residues padding. Due to we consider only the inference phase """
    N_TEMPL = "n_templ"
    N_MSA_SEQ = "n_msa_seq"
    N_EXTRA_SEQ = "n_extra_seq"
    N_RES = "n_res"

    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)
        self._shape_schema = {
            "template_aatype": [self.N_TEMPL, self.N_RES],
            "template_all_atom_mask": [self.N_TEMPL, self.N_RES, None],
            "template_all_atom_positions":
            [self.N_TEMPL, self.N_RES, None, None],
            "template_sum_probs": [self.N_TEMPL, None],
            "template_mask": [self.N_TEMPL],
            "template_pseudo_beta": [self.N_TEMPL, self.N_RES, None],
            "template_pseudo_beta_mask": [self.N_TEMPL, self.N_RES],
            "template_torsion_angles_sin_cos":
            [self.N_TEMPL, self.N_RES, None, None],
            "template_alt_torsion_angles_sin_cos":
            [self.N_TEMPL, self.N_RES, None, None],
            "template_torsion_angles_mask": [self.N_TEMPL, self.N_RES, None],
            "msa_mask": [self.N_MSA_SEQ, self.N_RES],
            "msa_row_mask": [self.N_MSA_SEQ],
            "extra_msa": [self.N_EXTRA_SEQ, self.N_RES],
            "extra_msa_mask": [self.N_EXTRA_SEQ, self.N_RES],
            "extra_msa_row_mask": [self.N_EXTRA_SEQ],
            "bert_mask": [self.N_MSA_SEQ, self.N_RES],
            "true_msa": [self.N_MSA_SEQ, self.N_RES],
            "extra_has_deletion": [self.N_EXTRA_SEQ, self.N_RES],
            "extra_deletion_value": [self.N_EXTRA_SEQ, self.N_RES],
            "msa_feat": [self.N_MSA_SEQ, self.N_RES, None],
        }
        self._pad_size_map = {
            self.N_TEMPL: self.config.max_templates,
            self.N_MSA_SEQ: self.config.max_msa_clusters,
            self.N_EXTRA_SEQ: self.config.max_extra_msa,
        }

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        for k, v in features.items():
            # Don't transfer this to the accelerator.
            if k == "extra_cluster_assignment":
                continue
            shape = list(v.shape)
            schema = self._shape_schema.get(k)
            if schema is None:
                continue
            msg = "Rank mismatch between shape and shape schema for"
            assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
            pad_size = [
                self._pad_size_map.get(s2, None) or s1
                for (s1, s2) in zip(shape, schema)
            ]

            padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
            padding.reverse()
            padding = list(itertools.chain(*padding))
            if padding:
                features[k] = torch.nn.functional.pad(v, padding)
                features[k] = torch.reshape(features[k], pad_size)

        return features
