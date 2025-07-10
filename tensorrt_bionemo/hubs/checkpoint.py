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

from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

from .hf import load_state_dict_from_hf

HFCheckpoint = namedtuple(
    "HFCheckpoint", ["repo_id", "filename", "weights_only", "state_dict_key"])

HF_CHECKPOINTS = {
    "boltz-1":
    HFCheckpoint(
        repo_id="boltz-community/boltz-1",
        filename="boltz1_conf.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    "boltz-2":
    HFCheckpoint(
        repo_id="boltz-community/boltz-2",
        filename="boltz2_conf.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    ),
    "boltz-2-affinity":
    HFCheckpoint(
        repo_id="boltz-community/boltz-2",
        filename="boltz2_aff.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    ),
}


def load_hf_weights(name: str,
                    local_files_only: bool = False,
                    cache_dir: Optional[Union[str, Path]] = None,
                    return_raw: bool = False,
                    repo_id: Optional[Union[str, Path]] = None):
    """ Load a checkpoint from the Hugging Face Hub """
    checkpoint = HF_CHECKPOINTS[name]
    default_repo_id = checkpoint.repo_id
    if repo_id is None:
        repo_id = default_repo_id
    return load_state_dict_from_hf(repo_id=repo_id,
                                   filename=checkpoint.filename,
                                   weights_only=checkpoint.weights_only,
                                   state_dict_key=checkpoint.state_dict_key,
                                   local_files_only=local_files_only,
                                   cache_dir=cache_dir,
                                   return_raw=return_raw)
