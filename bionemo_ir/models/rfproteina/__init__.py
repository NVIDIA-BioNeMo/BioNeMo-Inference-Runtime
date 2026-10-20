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
from bionemo_ir._torch.modules.rfproteina import (
    RFProteinaAtomizedOutput,
    RFProteinaCleanStateEncoding,
    RFProteinaConversionReport,
    RFProteinaCorruptionOutput,
    RFProteinaInsertionOutput,
    RFProteinaLatentEncoder,
    RFProteinaMaterializedOutput,
    RFProteinaNucleicAcidOutput,
    RFProteinaPartialRedesignOutput,
    RFProteinaPartialSamplingOutput,
    RFProteinaPredictionOutput,
    RFProteinaRepresentationBuilder,
    RFProteinaRepresentations,
    RFProteinaSamplingOutput,
    RFProteinaSearchCandidate,
    RFProteinaSearchCost,
    RFProteinaSearchOutput,
    apply_insertion_assignment,
    assign_unindexed_tokens,
    materialize_atomized_output,
    materialize_nucleic_acid_output,
)
from bionemo_ir._torch.modules.rfproteina.metrics import RFProteinaLocalMetricEvaluator
from bionemo_ir.hubs import resolve_rfproteina_autoencoder_checkpoint, resolve_rfproteina_checkpoint

from .config import (
    INVERSE_FOLDING_METHODS,
    RESAMPLED_FORWARD_MARGINAL,
    RFProteinaBeamSearchConfig,
    RFProteinaConfig,
    RFProteinaDecoderConfig,
    RFProteinaEncoderConfig,
    RFProteinaMetricClipConfig,
    RFProteinaModalitySamplingConfig,
    RFProteinaNoiseScheduleConfig,
    RFProteinaReferenceConfig,
    RFProteinaPartialSimulationConfig,
    RFProteinaSamplingConfig,
    RECYCLE_SCHEDULES,
    recycle_schedule,
    RFProteinaSearchRewardConfig,
    RFProteinaSearchRewardOverrideConfig,
    RFProteinaRepresentationConfig,
    RFProteinaScheduleConfig,
    RFProteinaTrunkConfig,
    RFProteinaVFStepConfig,
)
from .convert import (
    convert_rfproteina_decoder_torch,
    convert_rfproteina_encoder_torch,
    convert_rfproteina_representation_builder_torch,
    convert_rfproteina_trunk_torch,
    load_rfproteina_checkpoint_state,
    load_rfproteina_decoder_checkpoint,
    load_rfproteina_encoder_checkpoint,
    load_rfproteina_representation_builder_checkpoint,
    load_rfproteina_trunk_checkpoint,
    rfproteina_decoder_config_from_checkpoint,
    rfproteina_decoder_config_from_weights,
    rfproteina_encoder_config_from_checkpoint,
    rfproteina_encoder_config_from_weights,
    rfproteina_representation_config_from_weights,
    rfproteina_trunk_config_from_checkpoint,
    rfproteina_trunk_config_from_weights,
)
from .modeling import MODEL_NAME, RFProteina
# isort: on

__all__ = [
    "INVERSE_FOLDING_METHODS",
    "MODEL_NAME",
    "RESAMPLED_FORWARD_MARGINAL",
    "RFProteina",
    "RFProteinaAtomizedOutput",
    "RFProteinaBeamSearchConfig",
    "RFProteinaConfig",
    "RFProteinaCleanStateEncoding",
    "RFProteinaConversionReport",
    "RFProteinaCorruptionOutput",
    "RFProteinaDecoderConfig",
    "RFProteinaEncoderConfig",
    "RFProteinaInsertionOutput",
    "RFProteinaLatentEncoder",
    "RFProteinaLocalMetricEvaluator",
    "RFProteinaMaterializedOutput",
    "RFProteinaMetricClipConfig",
    "RFProteinaModalitySamplingConfig",
    "RFProteinaNucleicAcidOutput",
    "RFProteinaNoiseScheduleConfig",
    "RFProteinaPartialRedesignOutput",
    "RFProteinaPartialSamplingOutput",
    "RFProteinaPartialSimulationConfig",
    "RFProteinaPredictionOutput",
    "RFProteinaReferenceConfig",
    "RFProteinaRepresentationBuilder",
    "RFProteinaRepresentationConfig",
    "RFProteinaRepresentations",
    "RFProteinaSamplingConfig",
    "RECYCLE_SCHEDULES",
    "recycle_schedule",
    "RFProteinaSamplingOutput",
    "RFProteinaScheduleConfig",
    "RFProteinaSearchCandidate",
    "RFProteinaSearchCost",
    "RFProteinaSearchOutput",
    "RFProteinaSearchRewardConfig",
    "RFProteinaSearchRewardOverrideConfig",
    "RFProteinaTrunkConfig",
    "RFProteinaVFStepConfig",
    "apply_insertion_assignment",
    "assign_unindexed_tokens",
    "convert_rfproteina_decoder_torch",
    "convert_rfproteina_encoder_torch",
    "convert_rfproteina_representation_builder_torch",
    "convert_rfproteina_trunk_torch",
    "load_rfproteina_checkpoint_state",
    "load_rfproteina_decoder_checkpoint",
    "load_rfproteina_encoder_checkpoint",
    "load_rfproteina_representation_builder_checkpoint",
    "load_rfproteina_trunk_checkpoint",
    "materialize_atomized_output",
    "materialize_nucleic_acid_output",
    "resolve_rfproteina_autoencoder_checkpoint",
    "resolve_rfproteina_checkpoint",
    "rfproteina_decoder_config_from_checkpoint",
    "rfproteina_decoder_config_from_weights",
    "rfproteina_encoder_config_from_checkpoint",
    "rfproteina_encoder_config_from_weights",
    "rfproteina_representation_config_from_weights",
    "rfproteina_trunk_config_from_checkpoint",
    "rfproteina_trunk_config_from_weights",
]
