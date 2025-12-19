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

import io
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.hubs.support_matrix import FoldingSupportMatrix as SupMat

HFCheckpoint = namedtuple(
    "HFCheckpoint", ["repo_id", "filename", "weights_only", "state_dict_key"])

HF_CHECKPOINTS = {
    SupMat.Boltz1:
    HFCheckpoint(
        repo_id="boltz-community/boltz-1",
        filename="boltz1_conf.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    SupMat.Boltz2:
    HFCheckpoint(
        repo_id="boltz-community/boltz-2",
        filename="boltz2_conf.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    SupMat.Boltz2Affinity:
    HFCheckpoint(
        repo_id="boltz-community/boltz-2",
        filename="boltz2_aff.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    SupMat.OpenFold2_FT2:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_2.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_FT3:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_3.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_FT4:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_4.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_FT5:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_5.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_NoTempl1:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_no_templ_1.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_NoTempl2:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_no_templ_2.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_NoTempl_PTM1:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_no_templ_ptm_1.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_PTM1:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_ptm_1.pt",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_PTM2:
    HFCheckpoint(
        repo_id="nz/OpenFold",
        filename="finetuning_ptm_2.pt",
        weights_only=True,
        state_dict_key=None,
    ),
}


def load_state_dict_from_hf(
        repo_id: str,
        filename: str,
        weights_only: bool = False,
        state_dict_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        return_raw: bool = False) -> Union[io.BytesIO, dict[str]]:
    """ Load a state dict from the Hugging Face Hub """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "hf"
    cached_file = hf_hub_download(repo_id=repo_id,
                                  filename=filename,
                                  cache_dir=cache_dir,
                                  local_files_only=local_files_only)
    if return_raw:
        return cached_file
    logger.debug(f"Loading state dict from {cached_file}")
    try:
        state_dict = torch.load(cached_file, weights_only=weights_only)
    except TypeError:
        state_dict = torch.load(cached_file,
                                weights_only=weights_only,
                                map_location="cpu")
    if state_dict_key is not None:
        state_dict = state_dict[state_dict_key]
    return state_dict


def load_hf_weights(
    name: str,
    return_raw: bool = False,
    local_files_only: bool = False,
    cache_path: Optional[Union[str, Path]] = None,
    repo_id: Optional[Union[str,
                            Path]] = None) -> Union[io.BytesIO, dict[str]]:
    """ Load a checkpoint from the Hugging Face Hub """
    assert name in HF_CHECKPOINTS, f"Checkpoint {name} not found in HF_CHECKPOINTS"
    checkpoint = HF_CHECKPOINTS[name]
    default_repo_id = checkpoint.repo_id
    if repo_id is None:
        repo_id = default_repo_id
    return load_state_dict_from_hf(repo_id=repo_id,
                                   filename=checkpoint.filename,
                                   weights_only=checkpoint.weights_only,
                                   state_dict_key=checkpoint.state_dict_key,
                                   local_files_only=local_files_only,
                                   cache_dir=cache_path,
                                   return_raw=return_raw)
