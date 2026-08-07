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

import argparse
import os

import torch
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_


def get_model_basename(model_path: str) -> str:
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def main(args):
    config = model_config(args.config_preset)
    model = AlphaFold(config)
    model_basename = get_model_basename(args.jax_path)
    model_version = "_".join(model_basename.split("_")[1:])
    import_jax_weights_(model, args.jax_path, version=model_version)
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{model_basename}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax_path", type=str, help="Path to JAX checkpoint file", default="params_model_1.npz")
    parser.add_argument("--config_preset", type=str, help="The corresponding config preset", default="model_1")
    parser.add_argument("--output_dir", type=str, help="Path for output directory", default="output")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
