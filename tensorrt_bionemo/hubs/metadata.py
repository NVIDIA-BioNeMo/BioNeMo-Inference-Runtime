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
"""Model metadata registry and helpers for downloading / preparing assets."""

import logging
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from huggingface_hub import hf_hub_download

from tensorrt_bionemo.hubs.support_matrix import FoldingSupportMatrix as SupMat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared utility helpers
# ---------------------------------------------------------------------------


def download_hf_file(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path],
    local_files_only: bool = False,
) -> str:
    """Download a single file from HuggingFace Hub. Returns the local path."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
    )


def extract_archive(
    archive_path: Union[str, Path],
    extract_to: Union[str, Path],
) -> Path:
    """Extract a tar archive into *extract_to* and return the extraction root.

    If the archive contains a single top-level directory, that directory path
    is returned (e.g. ``extract_to/mols``).  Otherwise ``extract_to`` itself
    is returned.  Extraction is skipped when the target already exists.
    """
    extract_to = Path(extract_to)
    with tarfile.open(str(archive_path), "r:*") as tar:
        top_entries = {m.name.split("/")[0] for m in tar.getmembers()}
        if len(top_entries) == 1:
            result_dir = extract_to / top_entries.pop()
        else:
            result_dir = extract_to

        if result_dir.exists() and any(result_dir.iterdir()):
            logger.debug(
                f"Archive already extracted at {result_dir}, skipping")
            return result_dir

        logger.info(f"Extracting {archive_path} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        tar.extractall(path=extract_to, filter="data")

    return result_dir


def resolve_from_env(env_var: str) -> Optional[str]:
    """Return the value of an environment variable, or None."""
    return os.getenv(env_var)


def metadata_cache_dir() -> Path:
    """Root scanned for staged metadata assets: ``TENSORRT_BIONEMO_METADATA`` or
    ``<CACHE_DIR>/metadata`` (where ``run_tests.sh`` stages them)."""
    override = os.getenv("TENSORRT_BIONEMO_METADATA")
    if override:
        return Path(override)
    import tensorrt_bionemo
    return tensorrt_bionemo.CACHE_DIR / "metadata"


def resolve_cached_metadata(env: str) -> Optional[str]:
    """Path of a metadata asset staged at ``<metadata_dir>/<env>`` (a file or
    directory, typically a symlink created by run_tests.sh), else None. The
    staged target already matches what the ``env`` override would point to — the
    raw file for plain assets, the extracted dir for archives — so it is returned
    as-is with no further prepare step."""
    staged = metadata_cache_dir() / env
    if staged.exists():
        return str(staged)
    return None


def get_model_cache_dir(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return (and create) the per-model cache directory."""
    if cache_dir is not None:
        d = Path(cache_dir)
    else:
        import tensorrt_bionemo
        d = tensorrt_bionemo.CACHE_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Built-in prepare functions
# ---------------------------------------------------------------------------


def _prepare_plain_file(downloaded_path: str, cache_dir: Path) -> str:
    """No-op prepare: just return the downloaded path as-is."""
    return downloaded_path


def _prepare_tar_archive(downloaded_path: str, cache_dir: Path) -> str:
    """Extract a tar archive into *cache_dir* and return the result directory."""
    return str(extract_archive(downloaded_path, cache_dir))


# ---------------------------------------------------------------------------
# MetadataFile & per-model registry
# ---------------------------------------------------------------------------


@dataclass
class MetadataFile:
    """Describes a single metadata asset needed by a model.

    Attributes:
        metadata_key: Dict key in the returned metadata (e.g. "ccd_path").
        repo_id: HuggingFace repo to download from.
        filename: File name inside the repo.
        env: Environment variable for local override.
        prepare: Callable ``(downloaded_path, cache_dir) -> resolved_path``
            that transforms the raw download into whatever the model expects.
            Defaults to returning the downloaded file path unchanged.
    """
    metadata_key: str
    repo_id: str
    filename: str
    env: str
    prepare: Callable[[str, Path], str] = field(default=_prepare_plain_file)


HF_MODEL_METADATA: dict[str, list[MetadataFile]] = {
    SupMat.Boltz1: [
        MetadataFile(
            metadata_key="ccd_path",
            repo_id="boltz-community/boltz-1",
            filename="ccd.pkl",
            env="BOLTZ_CCD_PATH",
        ),
        # Boltz-1 reuses the Boltz-2 mols.tar: the inherited
        # Boltz2ContextGenerator.__call__ path always loads per-CCD-residue
        # molecule pickles via _get_molecules(mol_names), regardless of model.
        MetadataFile(
            metadata_key="mol_dir",
            repo_id="boltz-community/boltz-2",
            filename="mols.tar",
            env="BOLTZ_MOL_DIR",
            prepare=_prepare_tar_archive,
        ),
    ],
    SupMat.Boltz2: [
        MetadataFile(
            metadata_key="ccd_path",
            repo_id="boltz-community/boltz-1",
            filename="ccd.pkl",
            env="BOLTZ_CCD_PATH",
        ),
        MetadataFile(
            metadata_key="mol_dir",
            repo_id="boltz-community/boltz-2",
            filename="mols.tar",
            env="BOLTZ_MOL_DIR",
            prepare=_prepare_tar_archive,
        ),
    ],
    SupMat.Boltz2Affinity: [
        MetadataFile(
            metadata_key="ccd_path",
            repo_id="boltz-community/boltz-1",
            filename="ccd.pkl",
            env="BOLTZ_CCD_PATH",
        ),
        MetadataFile(
            metadata_key="mol_dir",
            repo_id="boltz-community/boltz-2",
            filename="mols.tar",
            env="BOLTZ_MOL_DIR",
            prepare=_prepare_tar_archive,
        ),
    ],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_metadata(
    model_name: str,
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Resolve metadata for a model, trying env vars first then HuggingFace.

    Each ``MetadataFile`` entry carries its own ``prepare`` callable, so the
    logic for turning a raw download into a usable path is fully customisable
    per-file.

    Returns a dict mapping metadata keys (e.g. "ccd_path", "mol_dir") to
    local filesystem paths.
    """
    if model_name not in HF_MODEL_METADATA:
        return {}

    resolved_cache_dir = get_model_cache_dir(model_name, cache_dir)

    metadata: dict[str, Any] = {}
    for meta_file in HF_MODEL_METADATA[model_name]:
        local_path = resolve_from_env(meta_file.env)
        if local_path is not None:
            logger.info(f"Using local {meta_file.metadata_key} from env "
                        f"{meta_file.env}: {local_path}")
            metadata[meta_file.metadata_key] = local_path
            continue

        # Fall back to an asset staged in the local cache (e.g. by run_tests.sh)
        # before giving up to the HuggingFace hub.
        staged = resolve_cached_metadata(meta_file.env)
        if staged is not None:
            logger.info(f"Using staged local {meta_file.metadata_key} for "
                        f"{model_name}: {staged}")
            metadata[meta_file.metadata_key] = staged
            continue

        logger.info(f"Downloading {meta_file.metadata_key} from "
                    f"{meta_file.repo_id}/{meta_file.filename}")
        downloaded = download_hf_file(
            meta_file.repo_id,
            meta_file.filename,
            resolved_cache_dir,
            local_files_only,
        )
        metadata[meta_file.metadata_key] = meta_file.prepare(
            downloaded, resolved_cache_dir)

    return metadata
