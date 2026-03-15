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

import random
from typing import Any, Callable, Optional

import numpy as np
import torch

from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import (FeatureCollatorBase,
                                            FeatureCollatorSpec,
                                            FeatureFactoryBase,
                                            FeatureGeneratorSpec,
                                            default_context_and_feature_merger)

from .feature_collators import (CropExtraMsa, DeleteExtraMsa, MakeFixedSize,
                                MakeMaskedMsa, MakeMsaFeat,
                                MultimerCreateMsaFeat, MultimerMakeMaskedMsa,
                                MultimerNearestNeighborClusters,
                                MultimerSampleMsa, NearestNeighborClusters,
                                RandomCropToSize, SampleMsa, SelectFeat,
                                SummarizeClusters)
from .feature_generators import (Atom37ToTorsionAngles, MakeAtom14Masks,
                                 MakeHhblitsProfile, MakeMsaMask,
                                 MakeSequenceMask, MakeTemplateMask,
                                 MakeTemplatePseudoBeta,
                                 MultimerCreateTargetFeatures,
                                 MultimerMakeMsaProfile, UseClampedFape)

_MONOMER_FEATURE_KEYS = [
    "aatype", "all_atom_mask", "all_atom_positions", "alt_chi_angles",
    "atom14_alt_gt_exists", "atom14_alt_gt_positions", "atom14_atom_exists",
    "atom14_atom_is_ambiguous", "atom14_gt_exists", "atom14_gt_positions",
    "atom37_atom_exists", "backbone_rigid_mask", "backbone_rigid_tensor",
    "bert_mask", "chi_angles_sin_cos", "chi_mask", "extra_deletion_value",
    "extra_has_deletion", "extra_msa", "extra_msa_mask", "extra_msa_row_mask",
    "is_distillation", "msa_feat", "msa_mask", "msa_row_mask",
    "no_recycling_iters", "pseudo_beta", "pseudo_beta_mask", "residue_index",
    "residx_atom14_to_atom37", "residx_atom37_to_atom14", "resolution",
    "rigidgroups_alt_gt_frames", "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous", "rigidgroups_gt_exists",
    "rigidgroups_gt_frames", "seq_length", "seq_mask", "target_feat",
    "template_aatype", "template_all_atom_mask", "template_all_atom_positions",
    "template_alt_torsion_angles_sin_cos", "template_backbone_rigid_mask",
    "template_backbone_rigid_tensor", "template_mask", "template_pseudo_beta",
    "template_pseudo_beta_mask", "template_sum_probs",
    "template_torsion_angles_mask", "template_torsion_angles_sin_cos",
    "true_msa", "use_clamped_fape", "is_template_present"
]

_MULTIMER_FEATURE_KEYS = [
    "aatype",
    "all_atom_mask",
    "all_atom_positions",
    # "all_chains_entity_ids",  # TODO: Resolve missing features, remove processed msa feats
    # "all_crops_all_chains_mask",
    # "all_crops_all_chains_positions",
    # "all_crops_all_chains_residue_ids",
    "assembly_num_chains",
    "asym_id",
    "atom14_atom_exists",
    "atom37_atom_exists",
    "bert_mask",
    "cluster_bias_mask",
    "cluster_profile",
    "cluster_deletion_mean",
    "deletion_matrix",
    "deletion_mean",
    "entity_id",
    "entity_mask",
    "extra_deletion_matrix",
    "extra_msa",
    "extra_msa_mask",
    # "mem_peak",
    "msa",
    "msa_feat",
    "msa_mask",
    "msa_profile",
    "num_alignments",
    "num_templates",
    # "queue_size",
    "residue_index",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "resolution",
    "seq_length",
    "seq_mask",
    "sym_id",
    "target_feat",
    "template_aatype",
    "template_all_atom_mask",
    "template_all_atom_positions",
    "true_msa",
    "is_template_present"
]


class SampleRepeater(FeatureCollatorBase):
    """
    A utility meta for feature collator. Call iteratively for n_times and concat the results.
    This useful when resampling with the same context on the same input batch.
    """

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 feature_collator_specs: list[FeatureCollatorSpec] = None,
                 get_n_iters: Optional[Callable] = None,
                 stack_dim: int = -1,
                 **kwargs):
        super().__init__(config, **kwargs)
        if get_n_iters is None:

            def get_n_iters(_: Optional[BaseConfig]) -> int:
                return 1

        self.n_iter = get_n_iters(config)
        self.stack_dim = stack_dim

        self.feature_collators = []
        if feature_collator_specs is not None:
            for v in feature_collator_specs:
                self.feature_collators.append(
                    v.functor(config=config, **v.kwargs))

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        ensemble_batch = []
        for _ in range(self.n_iter):
            batch_i = batch.copy()
            for collator in self.feature_collators:
                if collator.is_enabled():
                    batch_i = collator(batch_i, context)
            ensemble_batch.append(batch_i)
        ret = {}
        keys = ensemble_batch[0].keys()
        for k in keys:
            ret[k] = torch.stack([batch_i[k] for batch_i in ensemble_batch],
                                 dim=self.stack_dim)
        return ret


def pre_init(context: dict[str, Any]) -> dict[str, Any]:
    """ Setup environment for the feature factory."""
    random_seed = context.get("random_seed", 0)
    if random_seed is None:
        random_seed = random.randrange(2**32)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    context["ensemble_seed"] = random.randint(0, torch.iinfo(torch.int32).max)
    return context


def create_ensemble_feature_collator() -> list[FeatureCollatorSpec]:
    feature_collator_specs = [
        FeatureCollatorSpec(name="sample_msa", functor=SampleMsa, kwargs={}),
        FeatureCollatorSpec(name="make_masked_msa",
                            functor=MakeMaskedMsa,
                            kwargs={}),
        FeatureCollatorSpec(name="nearest_neighbor_clusters",
                            functor=NearestNeighborClusters,
                            kwargs={}),
        FeatureCollatorSpec(name="summarize_clusters",
                            functor=SummarizeClusters,
                            kwargs={}),
        FeatureCollatorSpec(name="crop_extra_msa",
                            functor=CropExtraMsa,
                            kwargs={}),
        FeatureCollatorSpec(name="delete_extra_msa",
                            functor=DeleteExtraMsa,
                            kwargs={}),
        FeatureCollatorSpec(name="make_msa_feat",
                            functor=MakeMsaFeat,
                            kwargs={}),
        FeatureCollatorSpec(name="select_feat",
                            functor=SelectFeat,
                            kwargs={"include_feats": _MONOMER_FEATURE_KEYS}),
        FeatureCollatorSpec(name="random_crop_to_size",
                            functor=RandomCropToSize,
                            kwargs={}),
        FeatureCollatorSpec(name="make_fixed_size",
                            functor=MakeFixedSize,
                            kwargs={}),
    ]
    return [
        FeatureCollatorSpec(name="repeater",
                            functor=SampleRepeater,
                            kwargs={
                                "feature_collator_specs":
                                feature_collator_specs,
                                "get_n_iters":
                                lambda config: config.max_recycling_iters + 1
                            })
    ]


class FeatureFactory(FeatureFactoryBase):
    # isort: off
    # How to debug the feature factory:
    # 1. OpenFold2 using some random in the feature generators.
    # 2. Set the fixed random seed in the feature factory at functor:
    #     - SampleMsa
    #     - MakeMaskedMsa
    #     - common.shaped_categorical
    #     - CropExtraMsa
    #     - transforms.RandomlyReplaceMsaWithUnknown
    # isort: on
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(name="use_clamped_fape",
                             functor=UseClampedFape,
                             kwargs={}),
        FeatureGeneratorSpec(name="make_sequence_mask",
                             functor=MakeSequenceMask,
                             kwargs={}),
        FeatureGeneratorSpec(name="make_msa_mask",
                             functor=MakeMsaMask,
                             kwargs={}),
        FeatureGeneratorSpec(name="make_hhblits_profile",
                             functor=MakeHhblitsProfile,
                             kwargs={}),
        FeatureGeneratorSpec(name="make_template_mask",
                             functor=MakeTemplateMask,
                             kwargs={}),
        FeatureGeneratorSpec(name="make_template_pseudo_beta",
                             functor=MakeTemplatePseudoBeta,
                             kwargs={}),
        FeatureGeneratorSpec(name="template_atom37_to_torsion_angles",
                             functor=Atom37ToTorsionAngles,
                             kwargs={"prefix": "template_"}),
        FeatureGeneratorSpec(name="make_atom14_masks",
                             functor=MakeAtom14Masks,
                             kwargs={}),
    ]
    features_merger_func: Callable = default_context_and_feature_merger
    feature_collator_specs: list[
        FeatureCollatorSpec] = create_ensemble_feature_collator()


def create_ensemble_multimer_feature_collator() -> list[FeatureCollatorSpec]:
    feature_collator_specs = [
        FeatureCollatorSpec(name="sample_msa",
                            functor=MultimerSampleMsa,
                            kwargs={}),
        FeatureCollatorSpec(name="make_masked_msa",
                            functor=MultimerMakeMaskedMsa,
                            kwargs={}),
        FeatureCollatorSpec(name="nearest_neighbor_clusters",
                            functor=MultimerNearestNeighborClusters,
                            kwargs={}),
        FeatureCollatorSpec(name="create_msa_feat",
                            functor=MultimerCreateMsaFeat,
                            kwargs={}),
        FeatureCollatorSpec(name="select_feat",
                            functor=SelectFeat,
                            kwargs={"include_feats": _MULTIMER_FEATURE_KEYS}),
        FeatureCollatorSpec(name="random_crop_to_size",
                            functor=RandomCropToSize,
                            kwargs={}),
        FeatureCollatorSpec(name="make_fixed_size",
                            functor=MakeFixedSize,
                            kwargs={}),
    ]
    return [
        FeatureCollatorSpec(name="repeater",
                            functor=SampleRepeater,
                            kwargs={
                                "feature_collator_specs":
                                feature_collator_specs,
                                "get_n_iters":
                                lambda config: config.max_recycling_iters + 1
                            })
    ]


class MultimerFeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        FeatureGeneratorSpec(name="make_msa_profile",
                             functor=MultimerMakeMsaProfile,
                             kwargs={}),
        FeatureGeneratorSpec(name="create_target_features",
                             functor=MultimerCreateTargetFeatures,
                             kwargs={}),
        FeatureGeneratorSpec(name="make_atom14_masks",
                             functor=MakeAtom14Masks,
                             kwargs={})
    ]
    features_merger_func: Callable = default_context_and_feature_merger
    feature_collator_specs: list[
        FeatureCollatorSpec] = create_ensemble_multimer_feature_collator()
