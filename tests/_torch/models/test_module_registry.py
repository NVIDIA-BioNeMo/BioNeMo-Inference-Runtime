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

"""Unit tests for ``ModuleRegistry``'s child-module pruning.

``_child_module_names`` reads each :class:`ModuleSpec`'s ``getter`` to recover
the module's attribute path (``lambda mod: mod.a.b`` -> ``("a", "b")``) and
returns the names whose path is a strict child of another spec's path. The
registry uses it to drop a child only when its parent is *also* requested. These
tests exercise both layers directly with lightweight specs — no model weights.
"""

import pytest

from tensorrt_bionemo.models.optimize_module_setter import (ModuleRegistry,
                                                            ModuleSpec,
                                                            _module_path)


def _spec(getter):
    """A ModuleSpec carrying only the getter under test (setter unused here)."""
    return ModuleSpec(getter=getter, setter=lambda mod, opt: None)


# --- _module_path ----------------------------------------------------------
def test_module_path_traces_attribute_chain():
    """``_module_path`` recovers the attribute chain a getter walks."""
    assert _module_path(_spec(lambda mod: mod.a.b.c)) == ("a", "b", "c")
    assert _module_path(_spec(lambda mod: mod.pairformer_stack)) == (
        "pairformer_stack", )


def test_non_attribute_getter_has_no_path():
    """Getters that index/call resolve to ``None`` (path undeterminable)."""
    assert _module_path(_spec(lambda mod: mod.blocks[0])) is None
    assert _module_path(_spec(lambda mod: mod.factory())) is None


# --- _child_module_names ---------------------------------------------------
def test_identifies_child_not_parent():
    """The nested spec is flagged; the parent is not."""
    specs = {
        # Mirrors the OpenFold3 registry: token_transformer lives *inside*
        # diffusion_module, so it is the child.
        "diffusion_module":
        _spec(lambda mod: mod.sample_diffusion.diffusion_module),
        "token_transformer":
        _spec(lambda mod: mod.sample_diffusion.diffusion_module.
              diffusion_transformer),
    }
    assert ModuleRegistry._child_module_names(specs) == {"token_transformer"}


def test_unrelated_siblings_have_no_children():
    """Disjoint module paths flag nothing."""
    specs = {
        "pairformer": _spec(lambda mod: mod.pairformer_stack),
        "msa": _spec(lambda mod: mod.trunk.msa_module),
        "structure": _spec(lambda mod: mod.structure_module.score_model),
    }
    assert ModuleRegistry._child_module_names(specs) == set()


def test_identical_paths_are_not_children():
    """Equal paths are not strict prefixes of each other, so neither flags."""
    specs = {
        "a": _spec(lambda mod: mod.trunk.block),
        "b": _spec(lambda mod: mod.trunk.block),
    }
    assert ModuleRegistry._child_module_names(specs) == set()


def test_untraceable_spec_is_neither_child_nor_parent():
    """A spec with no determinable path is never flagged nor used as parent."""
    specs = {
        "parent": _spec(lambda mod: mod.trunk),
        "child": _spec(lambda mod: mod.trunk.block),
        "weird": _spec(lambda mod: mod.blocks[0]),  # path -> None
    }
    # child flagged under parent; weird ignored because its path is unknown.
    assert ModuleRegistry._child_module_names(specs) == {"child"}


def test_deeply_nested_grandchild_is_a_child():
    """Prefix match applies at any depth, not just one level down."""
    specs = {
        "root": _spec(lambda mod: mod.a),
        "grandchild": _spec(lambda mod: mod.a.b.c.d),
    }
    assert ModuleRegistry._child_module_names(specs) == {"grandchild"}


def test_prefix_name_collision_is_not_a_child():
    """A shared attribute *prefix* (``block`` vs ``block_2``) isn't nesting."""
    # ("trunk", "block") is NOT a prefix of ("trunk", "block_2") as a path
    # tuple, so these are siblings, not parent/child.
    specs = {
        "block": _spec(lambda mod: mod.trunk.block),
        "block_2": _spec(lambda mod: mod.trunk.block_2),
    }
    assert ModuleRegistry._child_module_names(specs) == set()


# --- registry-level pruning (only prune when both are requested) -----------
class _FakeRegistry(ModuleRegistry):
    """Registry with a parent module and a child nested inside it."""

    def get_accelerated_modules(self):
        return {
            "diffusion_module":
            _spec(lambda mod: mod.sample_diffusion.diffusion_module),
            "token_transformer":
            _spec(lambda mod: mod.sample_diffusion.diffusion_module.
                  diffusion_transformer),
            "pairformer": _spec(lambda mod: mod.pairformer_stack),
        }


def _cfg():
    """Minimal placeholder config value (kept as-is by the registry)."""
    return object()


def test_child_alone_is_kept():
    """Requesting only the child (parent absent) keeps it accelerable."""
    reg = _FakeRegistry({"token_transformer": _cfg()})
    assert reg.get_module_names() == ["token_transformer"]


def test_child_dropped_when_parent_also_requested():
    """Requesting both parent and child drops the child in favour of parent."""
    reg = _FakeRegistry({
        "diffusion_module": _cfg(),
        "token_transformer": _cfg(),
    })
    assert reg.get_module_names() == ["diffusion_module"]


def test_unrelated_module_unaffected():
    """A sibling requested alongside the child is untouched."""
    reg = _FakeRegistry({
        "token_transformer": _cfg(),
        "pairformer": _cfg(),
    })
    assert set(reg.get_module_names()) == {"token_transformer", "pairformer"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
