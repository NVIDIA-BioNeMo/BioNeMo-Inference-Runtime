# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from pathlib import Path

import torch
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.main import BoltzProcessedInput
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo._torch.modules.boltz.physical.steering import \
    BoltzSteeringParams
from tensorrt_bionemo.models.boltz1 import Boltz1
from tensorrt_bionemo.models.optimize_module_setter import AcceleratedConfig
from tensorrt_bionemo.runtime import BackendType, OnDemandContextMemoryManager


class PathSetterMixin:
    """
    Mixin class to set data and output directories for Boltz writers.
    """

    def set_data_dir(self, data_dir: Path):
        self.data_dir = data_dir

    def set_output_dir(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


class GeneralBoltzWriter(BoltzWriter, PathSetterMixin):
    """
    General Boltz writer that can be used for the structure prediction.
    """

    def __init__(self,
                 data_dir: str = "/tmp/boltz_writer",
                 output_dir: str = "/tmp/boltz_output",
                 output_format: str = "mmcif",
                 boltz2: bool = False):
        BoltzWriter.__init__(self, data_dir, output_dir, output_format, boltz2)

    def set_output_format(self, output_format: str):
        self.output_format = output_format


def squeeze_output_dict(output: dict):
    """
    Squeeze the output dictionary to remove the batch dimension.
    """
    pair_chains_iptm = output.pop("pair_chains_iptm")
    ret = {
        k: v.squeeze(0)
        for k, v in output.items() if k not in {"masks", "token_masks"}
    }
    ret["masks"] = output["masks"]
    ret["token_masks"] = output.get("token_masks")
    ret["pair_chains_iptm"] = {}
    for chain1 in pair_chains_iptm.keys():
        ret["pair_chains_iptm"][chain1] = {}
        for chain2 in pair_chains_iptm[chain1].keys():
            ret["pair_chains_iptm"][chain1][chain2] = pair_chains_iptm[chain1][
                chain2].squeeze(0)
    return ret


def write_cif(writer, model, batch: dict, output: dict, output_dir: Path,
              dataloader_idx: int):
    writer.set_output_dir(output_dir)
    writer.write_on_batch_end(None,
                              model,
                              prediction=output,
                              batch_indices=[0],
                              batch=batch,
                              batch_idx=0,
                              dataloader_idx=dataloader_idx)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--structure_pairformer_ckpt',
        type=str,
        default="engines/structure_pairformer",
        help=
        'The path to the directory containing the structure pairformer engines'
    )
    parser.add_argument(
        '--confidence_pairformer_ckpt',
        type=str,
        default="engines/confidence_pairformer",
        help=
        'The path to the directory containing the confidence pairformer engines'
    )
    parser.add_argument(
        '--token_transformer_ckpt',
        type=str,
        default="engines/token_transformer",
        help=
        'The path to the directory containing the token transformer engines')
    parser.add_argument('--structure_pairformer_backend',
                        type=str,
                        default=BackendType.TORCH,
                        help='The backend to use for the structure pairformer')
    parser.add_argument(
        '--confidence_pairformer_backend',
        type=str,
        default=BackendType.TORCH,
        help='The backend to use for the confidence pairformer')
    parser.add_argument('--token_transformer_backend',
                        type=str,
                        default=BackendType.TORCH,
                        help='The backend to use for the token transformer')
    parser.add_argument(
        '--processed_dir',
        type=Path,
        default='boltz_processed_data',
        help='The path to the directory containing the processed data')
    parser.add_argument('--cache_dir',
                        type=Path,
                        default='.cache',
                        help='The path to the directory containing the cache')
    parser.add_argument('--gpu_per_node',
                        type=int,
                        default=8,
                        help='The number of GPUs per node')
    return parser.parse_args()


def run_single_rank(model: Boltz1,
                    processed_dir: Path,
                    cache_dir: Path,
                    num_workers: int = 1,
                    rank: int = 0,
                    device: torch.device = torch.device("cuda")):
    # TODO: run in parallel
    manifest = Manifest.load(processed_dir / "manifest.json")
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints") if
        (processed_dir / "constraints").exists() else None,
        template_dir=(processed_dir / "templates") if
        (processed_dir / "templates").exists() else None,
        extra_mols_dir=(processed_dir / "mols") if
        (processed_dir / "mols").exists() else None,
    )
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
        constraints_dir=processed.constraints_dir,
    )

    writer = GeneralBoltzWriter(boltz2=False)
    writer.set_data_dir(processed.targets_dir)
    stats = {}

    for idx, batch in enumerate(data_module.predict_dataloader()):
        record_id = batch["record"][0].id
        _, N_atoms, N_tokens = batch["atom_to_token"].shape
        batch = data_module.transfer_batch_to_device(batch, device, idx)
        stats[record_id] = {"n_tokens": N_tokens, "n_atoms": N_atoms}
        # Run optimized model
        with torch.no_grad():
            # try:
            torch.cuda.synchronize()
            start = time.time()
            output = model(feed_dict=batch,
                           recycling_steps=3,
                           num_sampling_steps=50,
                           diffusion_samples=1,
                           max_parallel_samples=None,
                           steering_args=BoltzSteeringParams())
            torch.cuda.synchronize()
            end = time.time()
            stats[record_id]["optimized_time"] = end - start
            output = squeeze_output_dict(output)
            print(
                f"Processed batch {record_id}, n_tokens: {N_tokens}, n_atoms: {N_atoms}, time: {end - start}"
            )
            # except Exception as e:
            #     print(f"Warning: Error running optimized model: {e}")
            #     stats[record_id]["error"] = f"optimized: {e}"
            #     continue
        output["exception"] = False
        write_cif(writer, model, batch, output, Path(f"output"), idx)

    return stats


def main(args):

    rank = 0
    torch.cuda.set_device(rank % args.gpu_per_node)

    logger.set_level("info")
    # dcp_size = 1  # FIXME: support dcp > 1

    # Create optimized model with TensorRT backends
    manager = OnDemandContextMemoryManager()
    model = Boltz1()
    model = model.cuda()
    model.load_weights()

    acc_m = {
        "structure_pairformer":
        AcceleratedConfig(
            checkpoint=args.structure_pairformer_ckpt,
            backend=args.structure_pairformer_backend,
        ),
        "confidence_pairformer":
        AcceleratedConfig(checkpoint=args.confidence_pairformer_ckpt,
                          backend=args.confidence_pairformer_backend),
        "token_transformer":
        AcceleratedConfig(checkpoint=args.token_transformer_ckpt,
                          backend=args.token_transformer_backend)
    }
    model = model.optimize(acc_m, manager)

    run_single_rank(processed_dir=args.processed_dir,
                    model=model,
                    rank=rank,
                    device=torch.device("cuda"),
                    cache_dir=args.cache_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
