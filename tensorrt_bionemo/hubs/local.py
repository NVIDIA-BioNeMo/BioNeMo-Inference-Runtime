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

import hashlib
import io
import logging
import os
import pickle
from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Optional, Union

import torch

from tensorrt_bionemo.hubs.support_matrix import FoldingSupportMatrix as SupMat

logger = logging.getLogger(__name__)

BOLTZ_MODEL_NAMES = frozenset({
    SupMat.Boltz1,
    SupMat.Boltz2,
    SupMat.Boltz2Affinity,
})
PROTENIX_MODEL_NAMES = frozenset({
    SupMat.ProtenixV2,
})

# MD5 digests of official HuggingFace Boltz checkpoints.
BOLTZ_CHECKPOINT_MD5 = {
    "boltz2_conf.ckpt": "2f0a1775bf8fc366a1a85e2019eca288",
    "boltz2_aff.ckpt": "8e93dadedd6edb7a4d170f6051b99ec0",
}

LocalCheckpoint = namedtuple("LocalCheckpoint",
                             ["env", "weights_only", "state_dict_key"])

LOCAL_CHECKPOINTS = {
    SupMat.Boltz1:
    LocalCheckpoint(
        env="BOLTZ1_CKPT",
        weights_only=True,
        state_dict_key="state_dict",
    ),
    SupMat.Boltz2:
    LocalCheckpoint(
        env="BOLTZ2_CKPT",
        weights_only=True,
        state_dict_key="state_dict",
    ),
    SupMat.Boltz2Affinity:
    LocalCheckpoint(
        env="BOLTZ2_AFFINITY_CKPT",
        weights_only=True,
        state_dict_key="state_dict",
    ),
    SupMat.OpenFold2_FT2:
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_FT3:
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_3_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_FT4:
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_4_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_FT5:
    LocalCheckpoint(
        env="OPENFOLD2_FINETUNING_5_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_NoTempl1:
    LocalCheckpoint(
        env="OPENFOLD2_NO_TEMPL_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_NoTempl2:
    LocalCheckpoint(
        env="OPENFOLD2_NO_TEMPL_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_NoTempl_PTM1:
    LocalCheckpoint(
        env="OPENFOLD2_NO_TEMPL_PTM_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_PTM1:
    LocalCheckpoint(
        env="OPENFOLD2_PTM_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold2_PTM2:
    LocalCheckpoint(
        env="OPENFOLD2_PTM_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_1:
    LocalCheckpoint(
        env="ALPHAFOLD2_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_2:
    LocalCheckpoint(
        env="ALPHAFOLD2_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_3:
    LocalCheckpoint(
        env="ALPHAFOLD2_3_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_4:
    LocalCheckpoint(
        env="ALPHAFOLD2_4_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_5:
    LocalCheckpoint(
        env="ALPHAFOLD2_5_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_Multimer_1:
    LocalCheckpoint(
        env="ALPHAFOLD2_MULTIMER_1_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_Multimer_2:
    LocalCheckpoint(
        env="ALPHAFOLD2_MULTIMER_2_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_Multimer_3:
    LocalCheckpoint(
        env="ALPHAFOLD2_MULTIMER_3_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_Multimer_4:
    LocalCheckpoint(
        env="ALPHAFOLD2_MULTIMER_4_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.AlphaFold2_Multimer_5:
    LocalCheckpoint(
        env="ALPHAFOLD2_MULTIMER_5_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.OpenFold3:
    LocalCheckpoint(
        env="OPENFOLD3_CKPT",
        weights_only=True,
        state_dict_key=None,
    ),
    SupMat.ProtenixV2:
    LocalCheckpoint(
        env="PROTENIX_V2_CKPT",
        weights_only=True,
        state_dict_key="model",
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
                                weights_only=True)
        if "ema" in state_dict:
            if "params" in state_dict["ema"]:
                state_dict = state_dict["ema"]["params"]
    return state_dict


def _load_boltz_state_dict(local_checkpoint: str):
    unsafe_globals = [
        (Dummy, 'omegaconf.base.ContainerMetadata'),
        (Dummy, 'omegaconf.base.Metadata'),
        (Dummy, 'omegaconf.dictconfig.DictConfig'),
        (Dummy, 'omegaconf.listconfig.ListConfig'),
        (Dummy, 'omegaconf.nodes.AnyNode'),
        (Dummy, 'typing.Any'),
        OrderedDict,
        defaultdict,
        int,
        bool,
        float,
        str,
        tuple,
        list,
        dict,
    ]
    with torch.serialization.safe_globals(unsafe_globals):
        return torch.load(local_checkpoint,
                          map_location="cpu",
                          weights_only=True)


def _load_protenix_state_dict(local_checkpoint: str):
    try:
        checkpoint = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=True)
    except (RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            "Failed to load Protenix checkpoint safely "
            "(unsafe or incompatible pickle contents).") from exc
    state_dict = checkpoint["model"]
    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def verify_boltz_checkpoint_md5(path: str | Path, filename: str) -> None:
    """Verify MD5 of a downloaded Boltz checkpoint against known digests."""
    expected = BOLTZ_CHECKPOINT_MD5.get(Path(filename).name)
    if expected is None:
        return
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"MD5 mismatch for {filename}: expected {expected}, got {actual}. "
            "The checkpoint file may be corrupted or tampered with.")


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
        logger.info(
            f"Not found local checkpoint for {name}, using default repo id {default_repo_id}"
        )
        return None
    logger.info(f"Loading {name} from local filesystem {filepath}")
    cached_file = open(filepath, 'rb')
    if return_raw:
        return cached_file
    cached_file.close()
    if name == SupMat.OpenFold3:
        state_dict = _load_of3_state_dict(filepath)
    elif name in BOLTZ_MODEL_NAMES:
        state_dict = _load_boltz_state_dict(filepath)
        if state_dict_key is not None:
            state_dict = state_dict[state_dict_key]
    elif name in PROTENIX_MODEL_NAMES:
        state_dict = _load_protenix_state_dict(filepath)
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
