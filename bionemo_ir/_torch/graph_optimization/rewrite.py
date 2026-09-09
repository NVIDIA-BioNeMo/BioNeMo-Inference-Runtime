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
"""Generic in-place module-tree rewrites."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch.nn as nn


def rewrite_modules[SourceT: nn.Module, TargetT: nn.Module](
    root: nn.Module,
    source_type: type[SourceT],
    replacement: Callable[[SourceT], TargetT],
    *,
    skip_types: tuple[type[nn.Module], ...] = (),
) -> int:
    """Replace every exact ``source_type`` descendant of ``root`` in place.

    This is an A-to-B tree rewrite: ``replacement`` receives each A and returns
    the B installed under the same parent name. Returning the source module
    leaves that child unchanged and continues into its descendants. Exact-type
    matching preserves subclasses with distinct semantics and makes a rewrite
    whose B subclasses A idempotent.

    The root itself is not replaced because a function cannot update the
    caller's reference. Subtrees rooted at ``skip_types`` are not inspected.
    Replacements are not traversed during the same pass.

    Args:
        root: Module whose descendants are rewritten.
        source_type: Exact child type to replace.
        replacement: Factory converting one source module into its replacement.
            Returning the source module is a no-op for that child.
        skip_types: Module types whose complete subtrees remain unchanged.

    Returns:
        Number of replaced modules.
    """
    count = 0
    for name, child in list(root.named_children()):
        if skip_types and isinstance(child, skip_types):
            continue
        if type(child) is source_type:
            new_child = replacement(cast(SourceT, child))
            if not isinstance(new_child, nn.Module):
                raise TypeError(
                    f"replacement for {source_type.__name__} returned {type(new_child).__name__}, expected nn.Module"
                )
            if new_child is child:
                count += rewrite_modules(
                    child,
                    source_type,
                    replacement,
                    skip_types=skip_types,
                )
            else:
                setattr(root, name, new_child)
                count += 1
        else:
            count += rewrite_modules(
                child,
                source_type,
                replacement,
                skip_types=skip_types,
            )
    return count
