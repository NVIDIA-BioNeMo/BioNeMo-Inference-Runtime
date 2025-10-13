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
import logging
import os
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

LocalCheckpoint = namedtuple("LocalCheckpoint",
                             ["env", "weights_only", "state_dict_key"])

LOCAL_CHECKPOINTS = {
    "boltz-1":
    LocalCheckpoint(
        env="BOLTZ1_CKPT",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    "boltz-2":
    LocalCheckpoint(
        env="BOLTZ2_CKPT",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    "boltz-2-affinity":
    LocalCheckpoint(
        env="BOLTZ2_AFFINITY_CKPT",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    "openfold2_finetuning_2":
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_finetuning_3":
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_3_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_finetuning_4":
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_4_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_finetuning_5":
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_5_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_no_templ_1":
    LocalCheckpoint(
        env="OPENFOLD2_NO_TEMPL_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_no_templ_2":
    LocalCheckpoint(
        env="OPENFOLD2_NO_TEMPL_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_no_templ_ptm_1":
    LocalCheckpoint(
        env="OPENFOLD2_NO_TEMPL_PTM_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_ptm_1":
    LocalCheckpoint(
        env="OPENFOLD2_PTM_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold2_ptm_2":
    LocalCheckpoint(
        env="OPENFOLD2_PTM_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    "openfold3":
    LocalCheckpoint(
        env="OPENFOLD3_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
}


class Dummy:

    def __init__(self, *args, **kwargs):
        pass


def _load_of3_state_dict(local_checkpoint: str):
    unsafe_globals = [
        (Dummy, 'openfold3.projects.of3_all_atom.model.OpenFold3'),
        (Dummy, 'ml_collections.config_dict.config_dict.ConfigDict'),
        (Dummy, 'ml_collections.config_dict.config_dict.FieldReference'),
        (Dummy, 'ml_collections.config_dict.config_dict._Op'),
        (Dummy, '_operator.add'),
        int,
        bool,
        float,
        str,
        tuple,
        list,
        dict,
    ]
    with torch.serialization.safe_globals(unsafe_globals):
        state_dict = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=True)["ema"]["params"]
    return state_dict


def load_local_weights(
        name: str,
        return_raw: bool = False,
        local_files_only: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        repo_id: Optional[Union[str,
                                Path]] = None) -> Union[io.BytesIO, dict[str]]:
    """ Load a checkpoint from the local filesystem """
    checkpoint = LOCAL_CHECKPOINTS[name]
    state_dict_key = checkpoint.state_dict_key
    weights_only = checkpoint.weights_only
    default_repo_id = checkpoint.env
    if cache_path is not None:
        filepath = cache_path
    else:
        if repo_id is not None:
            default_repo_id = repo_id
        filepath = os.getenv(default_repo_id)
    if filepath is None:
        return None
    logger.info(f"Loading {name} from local filesystem {filepath}")
    cached_file = open(filepath, 'rb')
    if return_raw:
        return cached_file

    if name == "openfold3":  # OpenFold3 has a different structure, we specifically extract the model weights
        state_dict = _load_of3_state_dict(filepath)
    else:
        try:
            state_dict = torch.load(filepath, weights_only=weights_only)
        except TypeError:
            state_dict = torch.load(filepath,
                                    weights_only=weights_only,
                                    map_location="cpu")
        if state_dict_key is not None:
            state_dict = state_dict[state_dict_key]
    return state_dict
