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
