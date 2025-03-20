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

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def load_state_dict_from_hf(repo_id: str,
                            filename: str,
                            weights_only: bool = False,
                            state_dict_key: Optional[str] = None,
                            cache_dir: Optional[Union[str, Path]] = None,
                            local_files_only: bool = False):
    """ Load a state dict from the Hugging Face Hub """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "hf"
    cached_file = hf_hub_download(repo_id=repo_id,
                                  filename=filename,
                                  cache_dir=cache_dir,
                                  local_files_only=local_files_only)
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
