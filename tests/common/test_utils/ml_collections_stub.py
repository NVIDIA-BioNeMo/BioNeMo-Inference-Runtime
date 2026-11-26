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
"""Minimal ``ml_collections.ConfigDict`` stand-in for OSS parity tests."""

from __future__ import annotations

import copy
import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


class ConfigDict(dict[str, Any]):
    """Nested attribute-accessible dictionary compatible with OSS ``ConfigDict`` usage."""

    def __init__(self, initial_dictionary: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if initial_dictionary is not None:
            self.update(initial_dictionary)
        if kwargs:
            self.update(kwargs)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to plain Python dictionaries."""
        return {k: v.to_dict() if isinstance(v, ConfigDict) else v for k, v in self.items()}

    def copy_and_resolve_references(self) -> ConfigDict:
        """Return a deep copy of this configuration dictionary."""
        return copy.deepcopy(self)


@contextmanager
def stub_ml_collections() -> Iterator[None]:
    """Provide ``ml_collections`` in ``sys.modules`` while importing OSS test layers.

    Leaves any already-loaded real ``ml_collections`` module untouched and restores
    ``sys.modules`` on exit.
    """
    names = ("ml_collections", "ml_collections.config_dict", "ml_collections.config_dict.config_dict")
    saved = {name: sys.modules.get(name) for name in names}
    had_keys = {name: name in sys.modules for name in names}

    if sys.modules.get("ml_collections") is None:
        pkg = types.ModuleType("ml_collections")
        pkg.__path__ = []
        pkg.ConfigDict = ConfigDict  # type: ignore[attr-defined]
        pkg.FrozenConfigDict = ConfigDict  # type: ignore[attr-defined]

        sub = types.ModuleType("ml_collections.config_dict")
        sub.__path__ = []
        sub.ConfigDict = ConfigDict  # type: ignore[attr-defined]
        sub.FrozenConfigDict = ConfigDict  # type: ignore[attr-defined]

        leaf = types.ModuleType("ml_collections.config_dict.config_dict")
        leaf.ConfigDict = ConfigDict  # type: ignore[attr-defined]
        leaf.FrozenConfigDict = ConfigDict  # type: ignore[attr-defined]

        pkg.config_dict = sub  # type: ignore[attr-defined]
        sub.config_dict = leaf  # type: ignore[attr-defined]
        sys.modules["ml_collections"] = pkg
        sys.modules["ml_collections.config_dict"] = sub
        sys.modules["ml_collections.config_dict.config_dict"] = leaf

    try:
        yield
    finally:
        for name in names:
            if had_keys[name]:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)
