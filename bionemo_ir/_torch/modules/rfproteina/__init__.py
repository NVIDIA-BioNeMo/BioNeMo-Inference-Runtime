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

# isort: off
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.normalization import AdaLN
from bionemo_ir._torch.layers.transformers.local_latents_decoder import RFProteinaLatentDecoder
from bionemo_ir._torch.layers.transformers.local_latents_encoder import RFProteinaLatentEncoder
from bionemo_ir._torch.layers.transformers.local_latents_transformer import (
    LocalLatentsTransformer,
    RFProteinaPairUpdate,
    RFProteinaTrunkLayer,
)
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock, Transition

from .atomized import ATOMIZED_RESIDUE_TYPE, ATOMIZED_TOKEN_TYPE, materialize_atomized_output
from .contracts import (
    ModalityReference,
    RFProteinaAtomizedOutput,
    RFProteinaCleanStateEncoding,
    RFProteinaConversionReport,
    RFProteinaCorruptionOutput,
    RFProteinaInsertionOutput,
    RFProteinaMaterializedOutput,
    RFProteinaNucleicAcidOutput,
    RFProteinaPartialRedesignOutput,
    RFProteinaPartialSamplingOutput,
    RFProteinaPredictFn,
    RFProteinaPrediction,
    RFProteinaPredictionOutput,
    RFProteinaRepresentationBuilderFn,
    RFProteinaRepresentations,
    RFProteinaSamplingOutput,
    RFProteinaSearchCandidate,
    RFProteinaSearchCost,
    RFProteinaSearchOutput,
    RFProteinaState,
)
from .flow_matcher import PRODUCT_SPACE_MODALITIES, RFProteinaProductSpaceFlowMatcher
from .groups import GROUP_KEYS, compose_group_ids, extract_groups
from .insertion import CONDITION_INDEX_MASK, UNASSIGNED_TOKEN_INDEX, apply_insertion_assignment, assign_unindexed_tokens
from .nucleic_acid import (
    DNA_TOKEN_TYPE,
    RNA_TOKEN_TYPE,
    materialize_nucleic_acid_output,
    validate_nucleic_acid_contract,
)
from .partial_redesign import materialize_partial_redesign_output
from .representation_builder import RFProteinaRepresentationBuilder
from .rollout import RFProteinaFlowMatchingIntegrator
from .sampler import RFProteinaPartialRolloutSetup, RFProteinaProductSpaceSampler, sample_reference_state
from .search import index_prediction, run_rfproteina_beam_search
from .schedules import (
    compute_tsr_ratio,
    create_fraction_partial_schedules,
    create_modality_schedule,
    create_partial_modality_schedule,
    create_partial_schedules,
    get_noise_rate,
    get_tsr_safe_schedule,
    partial_num_steps,
    resolve_partial_times,
)
from .vf_tsr import score_to_velocity, velocity_flow_step, velocity_to_clean, velocity_to_score

# isort: on
__all__ = [
    "AdaLN",
    "ATOMIZED_RESIDUE_TYPE",
    "ATOMIZED_TOKEN_TYPE",
    "AttentionPairBias",
    "ConditionedTransitionBlock",
    "CONDITION_INDEX_MASK",
    "DNA_TOKEN_TYPE",
    "GROUP_KEYS",
    "LocalLatentsTransformer",
    "ModalityReference",
    "PRODUCT_SPACE_MODALITIES",
    "RNA_TOKEN_TYPE",
    "RFProteinaFlowMatchingIntegrator",
    "RFProteinaAtomizedOutput",
    "RFProteinaCorruptionOutput",
    "RFProteinaCleanStateEncoding",
    "RFProteinaConversionReport",
    "RFProteinaInsertionOutput",
    "RFProteinaLatentDecoder",
    "RFProteinaLatentEncoder",
    "RFProteinaNucleicAcidOutput",
    "RFProteinaMaterializedOutput",
    "RFProteinaPartialRedesignOutput",
    "RFProteinaPartialRolloutSetup",
    "RFProteinaPartialSamplingOutput",
    "RFProteinaPairUpdate",
    "RFProteinaPredictFn",
    "RFProteinaPrediction",
    "RFProteinaPredictionOutput",
    "RFProteinaProductSpaceFlowMatcher",
    "RFProteinaProductSpaceSampler",
    "RFProteinaRepresentationBuilder",
    "RFProteinaRepresentationBuilderFn",
    "RFProteinaRepresentations",
    "RFProteinaSamplingOutput",
    "RFProteinaSearchCandidate",
    "RFProteinaSearchCost",
    "RFProteinaSearchOutput",
    "RFProteinaState",
    "RFProteinaTrunkLayer",
    "Transition",
    "UNASSIGNED_TOKEN_INDEX",
    "apply_insertion_assignment",
    "assign_unindexed_tokens",
    "compose_group_ids",
    "compute_tsr_ratio",
    "create_modality_schedule",
    "create_fraction_partial_schedules",
    "create_partial_modality_schedule",
    "create_partial_schedules",
    "extract_groups",
    "get_noise_rate",
    "get_tsr_safe_schedule",
    "materialize_atomized_output",
    "materialize_nucleic_acid_output",
    "materialize_partial_redesign_output",
    "index_prediction",
    "partial_num_steps",
    "resolve_partial_times",
    "run_rfproteina_beam_search",
    "sample_reference_state",
    "score_to_velocity",
    "velocity_flow_step",
    "velocity_to_clean",
    "velocity_to_score",
    "validate_nucleic_acid_contract",
]
