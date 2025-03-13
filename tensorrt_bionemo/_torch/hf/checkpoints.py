from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

from ._hub import load_state_dict_from_hf

HFCheckpoint = namedtuple(
    "HFCheckpoint", ["repo_id", "filename", "weights_only", "state_dict_key"])

HF_CHECKPOINTS = {
    "boltz-1":
    HFCheckpoint(
        repo_id="boltz-community/boltz-1",
        filename="boltz1_conf.ckpt",
        weights_only=False,
        state_dict_key="state_dict",
    )
}


def load_hf_weights(name: str,
                    local_files_only: bool = True,
                    cache_dir: Optional[Union[str, Path]] = None):
    """ Load a checkpoint from the Hugging Face Hub """
    checkpoint = HF_CHECKPOINTS[name]
    return load_state_dict_from_hf(repo_id=checkpoint.repo_id,
                                   filename=checkpoint.filename,
                                   weights_only=checkpoint.weights_only,
                                   state_dict_key=checkpoint.state_dict_key,
                                   local_files_only=local_files_only,
                                   cache_dir=cache_dir)
