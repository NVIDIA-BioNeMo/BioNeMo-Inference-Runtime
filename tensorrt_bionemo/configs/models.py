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
import torch
from transformers import PretrainedConfig

from tensorrt_bionemo.hubs.checkpoint import load_hf_weights

from .modules import PairformerConfig, TokenTransformerConfig


class Boltz1Config(PretrainedConfig):
    model_type = "boltz1"

    def __init__(self,
                 token_s: int = 384,
                 token_z: int = 128,
                 msa_pairwise_head_width: int = 32,
                 msa_pairwise_num_heads: int = 4,
                 pairformer_num_blocks: int = 48,
                 pairformer_num_heads: int = 16,
                 pairformer_no_update_s: bool = False,
                 pairformer_no_update_z: bool = False,
                 pairformer_dtype: str = "float32",
                 pairformer_backend_kwargs: dict = None,
                 token_transformer_num_blocks: int = 24,
                 token_transformer_heads: int = 16,
                 token_transformer_dtype: str = "float32",
                 token_transformer_backend_kwargs: dict = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.token_s = token_s
        self.token_z = token_z

        pairformer_backend_kwargs = pairformer_backend_kwargs or {}
        pairformer_backend_kwargs["token_s"] = token_s
        pairformer_backend_kwargs["token_z"] = token_z
        pairformer_backend_kwargs[
            "pairwise_head_width"] = msa_pairwise_head_width
        pairformer_backend_kwargs["pairwise_num_heads"] = msa_pairwise_num_heads
        pairformer_backend_kwargs["num_blocks"] = pairformer_num_blocks
        pairformer_backend_kwargs["num_heads"] = pairformer_num_heads
        pairformer_backend_kwargs["version"] = "v1"

        # TODO: Separate config for structure and confidence pairformer
        self.structure_pairformer_backend_config = PairformerConfig(
            architecture="structure_pairformer",
            dtype=pairformer_dtype,
            **pairformer_backend_kwargs)
        self.confidence_pairformer_backend_config = PairformerConfig(
            architecture="confidence_pairformer",
            dtype=pairformer_dtype,
            **pairformer_backend_kwargs)

        token_transformer_backend_kwargs = token_transformer_backend_kwargs or {}
        token_transformer_backend_kwargs[
            "num_blocks"] = token_transformer_num_blocks
        token_transformer_backend_kwargs["num_heads"] = token_transformer_heads
        token_transformer_backend_kwargs["dim"] = 2 * token_s
        token_transformer_backend_kwargs["dim_single_cond"] = 2 * token_s
        token_transformer_backend_kwargs["dim_pairwise"] = token_z
        token_transformer_backend_kwargs["version"] = "v1"

        self.token_transformer_backend_config = TokenTransformerConfig(
            architecture="token_transformer",
            dtype=token_transformer_dtype,
            **token_transformer_backend_kwargs)

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        **kwargs):
        if checkpoint_dir is None:
            ckpt = load_hf_weights(name="boltz-1", return_raw=True)
            state_dict = torch.load(ckpt,
                                    map_location="cpu",
                                    weights_only=False)
        else:
            state_dict = torch.load(checkpoint_dir,
                                    map_location="cpu",
                                    weights_only=False)
        hparams = state_dict["hyper_parameters"]
        return cls(
            token_s=hparams["token_s"],
            token_z=hparams["token_z"],
            msa_pairwise_head_width=hparams["msa_args"]["pairwise_head_width"],
            msa_pairwise_num_heads=hparams["msa_args"]["pairwise_num_heads"],
            pairformer_num_blocks=hparams["pairformer_args"]["num_blocks"],
            pairformer_num_heads=hparams["pairformer_args"]["num_heads"],
            token_transformer_num_blocks=hparams["score_model_args"]
            ["token_transformer_depth"],
            token_transformer_heads=hparams["score_model_args"]
            ["token_transformer_heads"],
            **kwargs)


class Boltz2Config(PretrainedConfig):
    model_type = "boltz2"

    def __init__(self,
                 token_s: int = 384,
                 token_z: int = 128,
                 msa_pairwise_head_width: int = 32,
                 msa_pairwise_num_heads: int = 4,
                 structure_pairformer_num_blocks: int = 64,
                 structure_pairformer_num_heads: int = 16,
                 confidence_pairformer_num_blocks: int = 8,
                 confidence_pairformer_num_heads: int = 16,
                 pairformer_dtype: str = "float32",
                 pairformer_backend_kwargs: dict = None,
                 post_layer_norm: bool = False,
                 token_transformer_num_blocks: int = 24,
                 token_transformer_heads: int = 16,
                 token_transformer_dtype: str = "float32",
                 token_transformer_backend_kwargs: dict = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.token_s = token_s
        self.token_z = token_z

        pairformer_backend_kwargs = pairformer_backend_kwargs or {}
        pairformer_backend_kwargs["token_s"] = token_s
        pairformer_backend_kwargs["token_z"] = token_z
        pairformer_backend_kwargs[
            "pairwise_head_width"] = msa_pairwise_head_width
        pairformer_backend_kwargs["pairwise_num_heads"] = msa_pairwise_num_heads
        pairformer_backend_kwargs[
            "num_blocks"] = structure_pairformer_num_blocks
        pairformer_backend_kwargs["num_heads"] = structure_pairformer_num_heads
        pairformer_backend_kwargs["version"] = "v2"
        pairformer_backend_kwargs["post_layer_norm"] = post_layer_norm

        # TODO: Separate config for structure and confidence pairformer
        self.structure_pairformer_backend_config = PairformerConfig(
            architecture="structure_pairformer",
            dtype=pairformer_dtype,
            **pairformer_backend_kwargs)
        pairformer_backend_kwargs[
            "num_blocks"] = confidence_pairformer_num_blocks
        pairformer_backend_kwargs["num_heads"] = confidence_pairformer_num_heads
        self.confidence_pairformer_backend_config = PairformerConfig(
            architecture="confidence_pairformer",
            dtype=pairformer_dtype,
            **pairformer_backend_kwargs)

        token_transformer_backend_kwargs = token_transformer_backend_kwargs or {}
        token_transformer_backend_kwargs[
            "num_blocks"] = token_transformer_num_blocks
        token_transformer_backend_kwargs["num_heads"] = token_transformer_heads
        token_transformer_backend_kwargs["dim"] = 2 * token_s
        token_transformer_backend_kwargs["dim_single_cond"] = 2 * token_s
        token_transformer_backend_kwargs["dim_pairwise"] = token_z
        token_transformer_backend_kwargs["version"] = "v2"

        self.token_transformer_backend_config = TokenTransformerConfig(
            architecture="token_transformer",
            dtype=token_transformer_dtype,
            **token_transformer_backend_kwargs)

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str = None,
                        trust_remote_code=False,
                        **kwargs):
        if checkpoint_dir is None:
            ckpt = load_hf_weights(name="boltz-2", return_raw=True)
            state_dict = torch.load(ckpt,
                                    map_location="cpu",
                                    weights_only=False)
        else:
            state_dict = torch.load(checkpoint_dir,
                                    map_location="cpu",
                                    weights_only=False)
        hparams = state_dict["hyper_parameters"]
        return cls(
            token_s=hparams["token_s"],
            token_z=hparams["token_z"],
            msa_pairwise_head_width=hparams["msa_args"]["pairwise_head_width"],
            msa_pairwise_num_heads=hparams["msa_args"]["pairwise_num_heads"],
            structure_pairformer_num_blocks=hparams["pairformer_args"]
            ["num_blocks"],
            structure_pairformer_num_heads=hparams["pairformer_args"]
            ["num_heads"],
            confidence_pairformer_num_blocks=hparams["confidence_model_args"]
            ["pairformer_args"]["num_blocks"],
            confidence_pairformer_num_heads=hparams["confidence_model_args"]
            ["pairformer_args"]["num_heads"],
            token_transformer_num_blocks=hparams["score_model_args"]
            ["token_transformer_depth"],
            token_transformer_heads=hparams["score_model_args"]
            ["token_transformer_heads"],
            **kwargs)
