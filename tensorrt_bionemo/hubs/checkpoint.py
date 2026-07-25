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

from pathlib import Path
from typing import Optional, Union

from tensorrt_bionemo.logger import logger

from .hf import load_hf_weights
from .local import load_local_weights


def load_weights(name: str,
                 return_raw: bool = False,
                 local_files_only: bool = False,
                 cache_path: Optional[Union[str, Path]] = None,
                 repo_id: Optional[Union[str, Path]] = None,
                 hub: str = None) -> Union[dict, str]:

    if hub is None:
        logger.warning(
            f"No hub specified, automatically trying local hub -> huggingface hub"
        )
        state_dict = load_local_weights(name, return_raw, local_files_only,
                                        cache_path, repo_id)
        if state_dict is None:
            state_dict = load_hf_weights(name, return_raw, local_files_only,
                                         None, repo_id)
        return state_dict
    elif hub == "local":
        state_dict = load_local_weights(name, return_raw, local_files_only,
                                        cache_path, repo_id)
        return state_dict
    elif hub == "hf":
        state_dict = load_hf_weights(name, return_raw, local_files_only, None,
                                     repo_id)
        return state_dict
    else:
        raise ValueError(f"Invalid hub: {hub}")
