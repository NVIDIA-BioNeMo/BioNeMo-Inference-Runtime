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

from pathlib import Path


def resolve_input_path(path: str | Path, allowed_root: str | Path | None = None) -> Path:
    """Resolve an input path and optionally confine it to a directory.

    Args:
        path: Input file path to resolve.
        allowed_root: Directory that must contain the resolved path. If unset,
            unrestricted local paths remain available for trusted callers.

    Returns:
        The resolved input path.

    Raises:
        ValueError: If ``allowed_root`` is not a directory, or if the resolved
            path is not a strict descendant of ``allowed_root``.
    """
    resolved_path = Path(path).resolve()
    if allowed_root is None:
        return resolved_path

    resolved_root = Path(allowed_root).resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"Allowed root {resolved_root!s} must be a directory")
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"Input path {path!s} is outside allowed root {resolved_root!s}")
    return resolved_path
