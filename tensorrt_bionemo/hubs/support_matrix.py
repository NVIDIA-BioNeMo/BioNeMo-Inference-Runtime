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


class FoldingSupportMatrix:
    Boltz1 = "boltz-1"
    Boltz2 = "boltz-2"
    Boltz2Affinity = "boltz-2-affinity"
    ProtenixV2 = "protenix-v2"
    OpenFold2_FT2 = "openfold2_finetuning_2"
    OpenFold2_FT3 = "openfold2_finetuning_3"
    OpenFold2_FT4 = "openfold2_finetuning_4"
    OpenFold2_FT5 = "openfold2_finetuning_5"
    OpenFold2_PTM1 = "openfold2_ptm_1"
    OpenFold2_PTM2 = "openfold2_ptm_2"
    OpenFold2_NoTempl1 = "openfold2_no_templ_1"
    OpenFold2_NoTempl2 = "openfold2_no_templ_2"
    OpenFold2_NoTempl_PTM1 = "openfold2_no_templ_ptm_1"
    AlphaFold2_1 = "alphafold2_1"
    AlphaFold2_2 = "alphafold2_2"
    AlphaFold2_3 = "alphafold2_3"
    AlphaFold2_4 = "alphafold2_4"
    AlphaFold2_5 = "alphafold2_5"
    AlphaFold2_Multimer_1 = "alphafold2_multimer_1"
    AlphaFold2_Multimer_2 = "alphafold2_multimer_2"
    AlphaFold2_Multimer_3 = "alphafold2_multimer_3"
    AlphaFold2_Multimer_4 = "alphafold2_multimer_4"
    AlphaFold2_Multimer_5 = "alphafold2_multimer_5"
    OpenFold3 = "openfold3"

    @staticmethod
    def is_supported(model_name: str) -> bool:
        return model_name in FoldingSupportMatrix.get_all_supported_model_names(
        )

    @staticmethod
    def get_all_supported_model_names() -> list[str]:
        return [
            value for key, value in FoldingSupportMatrix.__dict__.items()
            if not key.startswith("_") and isinstance(value, str)
        ]
