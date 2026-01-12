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

import random
from typing import Any, Callable, Optional

import numpy as np
import torch

from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import (FeatureCollatorBase,
                                            FeatureCollatorSpec,
                                            FeatureFactoryBase,
                                            FeatureGeneratorSpec,
                                            dict_context_merger)

from .feature_collators import (CropExtraMsa, DeleteExtraMsa, MakeFixedSize,
                                MakeMaskedMsa, MakeMsaFeat,
                                NearestNeighborClusters, RandomCropToSize,
                                SampleMsa, SelectFeat, SummarizeClusters)
from .feature_generators import (Atom37ToTorsionAngles, MakeAtom14Masks,
                                 MakeHhblitsProfile, MakeMsaMask,
                                 MakeSequenceMask, MakeTemplateMask,
                                 MakeTemplatePseudoBeta, UseClampedFape)

# isort: off
"""
How to debug the feature factory:
1. OpenFold2 using some random in the feature generators.
2. Set the fixed random seed in the feature factory at functor:
    - SampleMsa
    - MakeMaskedMsa
    - common.shaped_categorical
    - CropExtraMsa
    - transforms.RandomlyReplaceMsaWithUnknown
"""
# isort: on


class SampleRepeater(FeatureCollatorBase):
    """
    A utility meta for feature collator. Call iteratively for n_times and concat the results.
    This useful when resampling with the same context on the same input batch.
    """

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 feature_collator_specs: list[FeatureCollatorSpec] = None,
                 get_n_iters: Optional[Callable] = None,
                 stack_dim: int = -1):
        super().__init__(config)
        if get_n_iters is None:
            get_n_iters = lambda config: 1
        self.n_iter = get_n_iters(config)
        self.stack_dim = stack_dim

        self.feature_collators = []
        for v in feature_collator_specs:
            self.feature_collators.append(v.functor(config=config, **v.kwargs))

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        ensemble_batch = []
        for i in range(self.n_iter):
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
    """ Setup environment for the feature factory. """
    random.randrange(2**32)
    # np.random.seed(random_seed)
    np.random.seed(42)
    # torch.manual_seed(random_seed + 1)
    torch.manual_seed(42)
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
        FeatureCollatorSpec(
            name="select_feat",
            functor=SelectFeat,
            kwargs={
                "exclude_feats": [
                    "between_segment_residues", "deletion_matrix", "msa",
                    "num_alignments", "use_clamped_fape", "hhblits_profile",
                    "extra_deletion_matrix", "extra_cluster_assignment",
                    "cluster_profile", "cluster_deletion_mean",
                    "hhblits_profile"
                ]
            }),
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
    features_merger_func: Callable = dict_context_merger
    feature_collator_specs: list[
        FeatureCollatorSpec] = create_ensemble_feature_collator()
