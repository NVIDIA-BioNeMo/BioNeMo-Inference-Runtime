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
"""OSS RFProteina reference modules loaded from ``3rdparty/RFD4-Proteina``.

Upstream ``rfproteina/__init__.py`` eagerly imports its full training, config,
and AtomWorks data stack, and ``feature_factory.py`` imports AtomWorks /
Lightning helpers that are never executed by the neural-network parity tests.
This loader injects a scoped synthetic ``rfproteina`` package and minimal stubs
for those non-model imports so the upstream PyTorch modules run using only
BioIR's existing dependencies (PyTorch, NumPy, Einops, SciPy).
"""

from __future__ import annotations

import importlib
import io
import logging
import pickle
import re
import sys
import types
import typing
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path

import pytest
import torch

import tests
from tests.common.test_utils.basic import path_for_package_in_repo

__all__ = [
    "load_rfproteina_checkpoint_for_oss",
    "load_upstream_encoder_transformer",
    "load_upstream_noise_alignment",
]

_VENDORED_SRC = path_for_package_in_repo(tests).parent / "3rdparty" / "RFD4-Proteina" / "src"
_INTERPOLATION_RE = re.compile(r"^\$\{(\.+)?([^}]+)\}$")

_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "long",
        "object",
        "range",
        "set",
        "slice",
        "str",
        "tuple",
        "unicode",
        "xrange",
    }
)

_WEIGHT_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_sparse_tensor"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
    }
)


class _JaxtypingStub:
    """Subscriptable placeholder for jaxtyping shape annotations."""

    def __class_getitem__(cls, _item: object) -> type[object]:
        return object


class _ConditionStub:
    """Minimal stand-in carrying ``full_name`` for unused feature-factory conditions."""

    def __init__(self, name: str) -> None:
        self.full_name = f"condition_{name}"


class _HBondEnum(IntEnum):
    NONE = 0
    DONOR = 1
    ACCEPTOR = 2
    BOTH = 3


class _PiCationEnum(IntEnum):
    NONE = 0
    AROMATIC = 1
    CATION = 2
    BOTH = 3


class _SaltBridgeEnum(IntEnum):
    NONE = 0
    POSITIVE = 1
    NEGATIVE = 2


class _MetalCoordEnum(IntEnum):
    NONE = 0
    METAL = 1
    COORDINATING = 2


class _RasaEnum(IntEnum):
    UNSPECIFIED = 0
    BURIED = 1
    INTERMEDIATE = 2
    EXPOSED = 3


class _TokenType:
    """Mirrors ``rfproteina.datapipes.schema.proteina.TokenType``."""

    PROTEIN = 0
    RNA = 1
    DNA = 2
    ATOMIZED = 3
    UNKNOWN = 4
    GLYCAN = 5
    NCAA = 6


class _RankedLogger(logging.LoggerAdapter):
    """Drop-in stand-in for ``rfproteina.utils.ddp.RankedLogger`` without Lightning."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(logging.getLogger(name), extra or {})
        self.rank_zero_only = rank_zero_only


def _make_package(name: str, path: list[str] | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else path
    module.__package__ = name
    return module


def _build_import_stubs(src_root: Path) -> dict[str, types.ModuleType]:
    """Build minimal ``sys.modules`` stubs for upstream non-model imports."""
    stubs: dict[str, types.ModuleType] = {}

    # Type-annotation-only and unused helper imports in the model closure.
    jaxtyping_mod = types.ModuleType("jaxtyping")
    for attr in ("Bool", "Float", "Int", "Integer", "Num", "Real", "Shaped", "UInt"):
        setattr(jaxtyping_mod, attr, _JaxtypingStub)
    stubs["jaxtyping"] = jaxtyping_mod

    beartype_mod = _make_package("beartype")
    beartype_mod.typing = typing
    stubs["beartype"] = beartype_mod
    stubs["beartype.typing"] = typing

    toolz_mod = types.ModuleType("toolz")
    toolz_mod.merge_with = None
    stubs["toolz"] = toolz_mod

    # AtomWorks condition symbols imported at top level by feature_factory.py.
    stubs["atomworks"] = _make_package("atomworks")
    stubs["atomworks.ml"] = _make_package("atomworks.ml")
    aw_conditions = types.ModuleType("atomworks.ml.conditions")
    aw_conditions.C_CRD = _ConditionStub("coord")
    aw_conditions.C_SEQ = _ConditionStub("seq")
    stubs["atomworks.ml.conditions"] = aw_conditions

    # Synthetic rfproteina root package: skips rfproteina/__init__.py side effects
    # (environs, datapipes, standard_annotations) while exposing real submodules.
    stubs["rfproteina"] = _make_package("rfproteina", [str(src_root / "rfproteina")])

    # Non-model helper modules imported at top level by rfproteina.nn.feature_factory.
    rfp_conditions = types.ModuleType("rfproteina.conditions")
    rfp_conditions.HBondEnum = _HBondEnum
    rfp_conditions.PiCationEnum = _PiCationEnum
    rfp_conditions.SaltBridgeEnum = _SaltBridgeEnum
    rfp_conditions.MetalCoordEnum = _MetalCoordEnum
    rfp_conditions.RasaEnum = _RasaEnum
    rfp_conditions.SEQUENCE_PROBABILITY_CONDITIONS = ()
    for condition_name in (
        "C_CATH",
        "C_CENTROID",
        "C_CRDGRP",
        "C_HB",
        "C_HOT",
        "C_HPHOB",
        "C_IDXGRP",
        "C_INTCRD",
        "C_MCOORD",
        "C_PICATION",
        "C_PISTACK",
        "C_RAS",
        "C_SALT",
        "C_SEC",
    ):
        setattr(rfp_conditions, condition_name, _ConditionStub(condition_name))
    stubs["rfproteina.conditions"] = rfp_conditions

    stubs["rfproteina.datapipes"] = _make_package("rfproteina.datapipes")
    stubs["rfproteina.datapipes.schema"] = _make_package("rfproteina.datapipes.schema")
    rfp_schema = types.ModuleType("rfproteina.datapipes.schema.proteina")
    rfp_schema.TokenType = _TokenType
    rfp_schema.NUM_TOKEN_TYPES = 7
    stubs["rfproteina.datapipes.schema.proteina"] = rfp_schema

    rfp_encoding = types.ModuleType("rfproteina.encoding")
    rfp_encoding.UNIFIED_ATOM37_ENCODING = None
    stubs["rfproteina.encoding"] = rfp_encoding

    stubs["rfproteina.transforms"] = _make_package("rfproteina.transforms")
    rfp_chirality = types.ModuleType("rfproteina.transforms.chirality_feature")
    rfp_chirality.MAX_OVERLAPPING_CENTERS = 6
    stubs["rfproteina.transforms.chirality_feature"] = rfp_chirality

    rfp_ddp = types.ModuleType("rfproteina.utils.ddp")
    rfp_ddp.RankedLogger = _RankedLogger
    stubs["rfproteina.utils.ddp"] = rfp_ddp

    return stubs


@contextmanager
def _isolated_rfproteina_imports(src_root: Path) -> Iterator[None]:
    """Temporarily stub non-model dependencies while importing vendored ``rfproteina``."""
    stubs = _build_import_stubs(src_root)
    saved_modules: dict[str, types.ModuleType | None] = {}
    absent_modules: set[str] = set()
    for name in stubs:
        if name in sys.modules:
            saved_modules[name] = sys.modules[name]
        else:
            absent_modules.add(name)
    saved_path = list(sys.path)
    # Record any pre-existing rfproteina.* modules outside stubs so we restore them cleanly.
    preexisting_rfp = {
        name: mod
        for name, mod in sys.modules.items()
        if (name == "rfproteina" or name.startswith("rfproteina.")) and name not in stubs
    }
    try:
        sys.modules.update(stubs)
        yield
    finally:
        for name in list(sys.modules):
            if (name == "rfproteina" or name.startswith("rfproteina.")) and name not in stubs:
                if name in preexisting_rfp:
                    sys.modules[name] = preexisting_rfp[name]
                else:
                    sys.modules.pop(name, None)
        for name in absent_modules:
            sys.modules.pop(name, None)
        for name, previous in saved_modules.items():
            sys.modules[name] = previous
        # rfproteina.utils.align mutates sys.path on import; always restore it.
        sys.path[:] = saved_path


def _import_upstream_module(module_name: str) -> types.ModuleType:
    """Import an upstream ``rfproteina`` submodule from the vendored checkout or environment."""
    pkg_dir = _VENDORED_SRC / "rfproteina"
    if pkg_dir.is_dir() and next(pkg_dir.iterdir(), None) is not None:
        with _isolated_rfproteina_imports(_VENDORED_SRC):
            return importlib.import_module(module_name)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pytest.skip(
            "the RFD4-Proteina reference implementation is not checked out; "
            "initialise it with `make -C docker submodules`"
        )


def load_upstream_noise_alignment() -> types.ModuleType:
    """Return upstream ``rfproteina.nn.noise_alignment`` without third-party training deps."""
    return _import_upstream_module("rfproteina.nn.noise_alignment")


def load_upstream_encoder_transformer() -> type[torch.nn.Module]:
    """Return upstream ``rfproteina.partial_autoencoder.encoder.EncoderTransformer``."""
    module = _import_upstream_module("rfproteina.partial_autoencoder.encoder")
    return module.EncoderTransformer


class _MetadataStub(dict):
    """Inert placeholder for Lightning training globals absent from the test env."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()

    def __setstate__(self, _state: object) -> None:
        return None

    def __call__(self, *_args: object, **_kwargs: object) -> _MetadataStub:
        return _MetadataStub()


class _OmegaContainer:
    """Lightweight stand-in for pickled ``omegaconf.DictConfig`` / ``ListConfig``."""

    def __init__(self) -> None:
        self._content: object = None

    def __setstate__(self, state: dict[str, object]) -> None:
        self._content = state.get("_content")


class _OmegaNode:
    """Lightweight stand-in for pickled ``omegaconf.nodes.*Node`` scalars."""

    def __init__(self) -> None:
        self._val: object = None

    def __setstate__(self, state: dict[str, object]) -> None:
        self._val = state.get("_val")


class _OmegaCheckpointUnpickler(pickle.Unpickler):
    """Unpickle tensors and OmegaConf hyperparameter containers without ``omegaconf``."""

    def find_class(self, module: str, name: str) -> object:
        if module in ("builtins", "__builtin__"):
            if name in _SAFE_BUILTINS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Disallowed builtin global {module}.{name}")
        if (module, name) in _WEIGHT_GLOBALS or (module == "torch" and name.endswith("Storage")):
            return getattr(importlib.import_module(module), name)
        if module in ("omegaconf.dictconfig", "omegaconf.listconfig") and name in ("DictConfig", "ListConfig"):
            return _OmegaContainer
        if module == "omegaconf.nodes":
            return _OmegaNode
        return _MetadataStub


class _OmegaPickleModule:
    """``pickle_module`` adapter for :func:`torch.load`."""

    Unpickler = _OmegaCheckpointUnpickler

    @staticmethod
    def load(file: io.IOBase, **kwargs: object) -> object:
        return _OmegaCheckpointUnpickler(file, **kwargs).load()


def _unwrap_omegaconf(obj: object) -> object:
    """Convert unpickled ``_OmegaContainer`` / ``_OmegaNode`` trees into plain Python types."""
    if isinstance(obj, _OmegaContainer):
        return _unwrap_omegaconf(obj._content)
    if isinstance(obj, _OmegaNode):
        return _unwrap_omegaconf(obj._val)
    if isinstance(obj, dict):
        return {key: _unwrap_omegaconf(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_omegaconf(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_unwrap_omegaconf(value) for value in obj)
    return obj


def _resolve_interpolations(obj: object, root: dict[str, object], parents: tuple[dict[str, object], ...]) -> object:
    """Resolve simple OmegaConf ``${...}`` key references against plain nested dicts."""
    if isinstance(obj, dict):
        next_parents = (*parents, obj)
        return {key: _resolve_interpolations(value, root, next_parents) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_interpolations(value, root, parents) for value in obj]
    if isinstance(obj, str):
        match = _INTERPOLATION_RE.match(obj)
        if match is not None:
            dots, dotted_key = match.groups()
            base: object = root if not dots else parents[-len(dots)]
            for part in dotted_key.split("."):
                if not isinstance(base, dict) or part not in base:
                    return obj
                base = base[part]
            return _resolve_interpolations(base, root, parents)
    return obj


def load_rfproteina_checkpoint_for_oss(checkpoint: str | Path) -> dict[str, object]:
    """Load an RFProteina Lightning checkpoint's ``state_dict`` and ``hyper_parameters`` without ``omegaconf``.

    Args:
        checkpoint: Path to a released RFProteina ``.ckpt`` file.

    Returns:
        Dictionary containing ``state_dict`` tensors and plain-Python
        ``hyper_parameters`` with OmegaConf containers unwrapped and simple
        ``${...}`` interpolations resolved.
    """
    raw = torch.load(
        str(checkpoint),
        map_location="cpu",
        mmap=True,
        weights_only=False,
        pickle_module=_OmegaPickleModule,
    )
    hyper_parameters = _unwrap_omegaconf(raw["hyper_parameters"])
    if isinstance(hyper_parameters, dict):
        hyper_parameters = _resolve_interpolations(hyper_parameters, hyper_parameters, ())
    return {
        "state_dict": raw["state_dict"],
        "hyper_parameters": hyper_parameters,
    }
