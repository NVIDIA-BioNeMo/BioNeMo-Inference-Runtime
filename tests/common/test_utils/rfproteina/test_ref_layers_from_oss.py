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
"""Unpickling tests for the vendored RFProteina OSS checkpoint loader."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from tests.common.test_utils.rfproteina.ref_layers_from_oss import load_rfproteina_checkpoint_for_oss


class _PicklableOmegaContainer:
    """Helper whose pickle global mimics ``omegaconf.{Dict,List}Config``."""

    def __init__(self, content: object) -> None:
        self._content = content

    def __getstate__(self) -> dict[str, object]:
        return {"_content": self._content, "_metadata": None, "_parent": None}


class _PicklableOmegaNode:
    """Helper whose pickle global mimics ``omegaconf.nodes.AnyNode``."""

    def __init__(self, val: object) -> None:
        self._val = val

    def __getstate__(self) -> dict[str, object]:
        return {"_val": self._val, "_metadata": None, "_parent": None}


def test_load_rfproteina_checkpoint_for_oss_unwraps_omegaconf_without_omegaconf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OmegaConf DictConfig/ListConfig/AnyNode and ``${..key}`` interpolations unwrap to plain dicts."""
    # Temporarily provide synthetic omegaconf.* classes only while writing the checkpoint.
    fake_dict_cls = type("DictConfig", (_PicklableOmegaContainer,), {"__module__": "omegaconf.dictconfig"})
    fake_list_cls = type("ListConfig", (_PicklableOmegaContainer,), {"__module__": "omegaconf.listconfig"})
    fake_node_cls = type("AnyNode", (_PicklableOmegaNode,), {"__module__": "omegaconf.nodes"})

    fake_root = types.ModuleType("omegaconf")
    fake_dict_mod = types.ModuleType("omegaconf.dictconfig")
    fake_dict_mod.DictConfig = fake_dict_cls
    fake_list_mod = types.ModuleType("omegaconf.listconfig")
    fake_list_mod.ListConfig = fake_list_cls
    fake_nodes_mod = types.ModuleType("omegaconf.nodes")
    fake_nodes_mod.AnyNode = fake_node_cls

    ckpt_path = tmp_path / "synthetic_ae.ckpt"
    with monkeypatch.context() as save_ctx:
        save_ctx.setitem(sys.modules, "omegaconf", fake_root)
        save_ctx.setitem(sys.modules, "omegaconf.dictconfig", fake_dict_mod)
        save_ctx.setitem(sys.modules, "omegaconf.listconfig", fake_list_mod)
        save_ctx.setitem(sys.modules, "omegaconf.nodes", fake_nodes_mod)
        payload = {
            "state_dict": {"encoder.weight": torch.tensor([1.0, 2.0])},
            "hyper_parameters": fake_dict_cls(
                {
                    "cfg_ae": fake_dict_cls(
                        {
                            "nn_ae": fake_dict_cls(
                                {
                                    "latent_z_dim": fake_node_cls(32),
                                    "encoder": fake_dict_cls(
                                        {
                                            "nlayers": fake_node_cls(12),
                                            "latent_z_dim": fake_node_cls("${..latent_z_dim}"),
                                            "feats_seq": fake_list_cls(
                                                [fake_node_cls("token_type"), fake_node_cls("chain_idx_seq")]
                                            ),
                                        }
                                    ),
                                }
                            )
                        }
                    )
                }
            ),
        }
        torch.save(payload, ckpt_path)

    # Now poison omegaconf completely before loading.
    for loaded_name in [name for name in sys.modules if name == "omegaconf" or name.startswith("omegaconf.")]:
        monkeypatch.setitem(sys.modules, loaded_name, None)
    monkeypatch.setitem(sys.modules, "omegaconf", None)

    loaded = load_rfproteina_checkpoint_for_oss(ckpt_path)
    nn_ae = loaded["hyper_parameters"]["cfg_ae"]["nn_ae"]
    assert nn_ae == {
        "latent_z_dim": 32,
        "encoder": {
            "nlayers": 12,
            "latent_z_dim": 32,
            "feats_seq": ["token_type", "chain_idx_seq"],
        },
    }
    torch.testing.assert_close(loaded["state_dict"]["encoder.weight"], torch.tensor([1.0, 2.0]))


def test_checkpoint_loader_rejects_unsafe_builtins(tmp_path: Path) -> None:
    """Disallowed builtins such as ``eval`` must fail unpickling."""
    import pickle

    class _Evil:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    ckpt_path = tmp_path / "unsafe.ckpt"
    torch.save({"payload": _Evil()}, ckpt_path)

    with pytest.raises(pickle.UnpicklingError, match=r"Disallowed builtin global (?:builtins|__builtin__)\.eval"):
        load_rfproteina_checkpoint_for_oss(ckpt_path)
