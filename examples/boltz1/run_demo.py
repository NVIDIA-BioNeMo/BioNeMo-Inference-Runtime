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
import glob
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import boltz.data.const as const
import pandas as pd
import torch
import torch.nn as nn
from boltz.data.feature.pad import pad_dim
from boltz.model.model import Boltz1
# isort: on
from pytorch_lightning import seed_everything
from score import kabsch_torch, lddt
from tensorrt_llm.logger import logger

from tensorrt_bionemo.confs.modules.transformers import PairformerConfig
from tensorrt_bionemo.hf.checkpoints import load_hf_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.convert import (convert_hf_pairformer,
                                                    torch_pairformer_load_fn)
from tensorrt_bionemo.modules.pairformer import PairformerBackendBuilder

SEED = 42
"""
NOTE:
    This script is used to run the demo of the Boltz1 model along with torch backbone from the original repo.
    It is used to verify the correctness of the TensorRT-BNM implementation. The inputs to model is dumped by `botlz predict`.
    For usage TRT-engines in production, please use _torch.backend for models.
"""


class ForceFP32(nn.Module):

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self._original_module = original_module

    def forward(self, *args, **kwargs):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        output = self._original_module(*args, **kwargs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return output


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class BoltzPredictionParams:
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False


def create_original_model(device: torch.device) -> nn.Module:
    cached_file = load_hf_weights("boltz-1", return_raw=True)
    predict_params = BoltzPredictionParams()
    diffusion_params = BoltzDiffusionParams()

    model: Boltz1 = Boltz1.load_from_checkpoint(
        cached_file,
        strict=True,
        predict_args=asdict(predict_params),
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )
    model.eval()
    return model, predict_params


def pad_batch(batch: dict, pad_seqlen: int, seqlen: int):
    pad_len = pad_seqlen - seqlen
    if pad_len == 0:
        return batch

    # Padding for msa features
    batch["msa"] = pad_dim(batch["msa"], 2, pad_len, const.token_ids["-"])
    batch["msa_paired"] = pad_dim(batch["msa_paired"], 2, pad_len)
    batch["deletion_value"] = pad_dim(batch["deletion_value"], 2, pad_len)
    batch["has_deletion"] = pad_dim(batch["has_deletion"], 2, pad_len)
    batch["deletion_mean"] = pad_dim(batch["deletion_mean"], 1, pad_len)
    batch["profile"] = pad_dim(batch["profile"], 1, pad_len)
    batch["msa_mask"] = pad_dim(batch["msa_mask"], 2, pad_len)

    # Padding for atom features
    batch["atom_to_token"] = pad_dim(batch["atom_to_token"], 2, pad_len)
    batch["token_to_rep_atom"] = pad_dim(batch["token_to_rep_atom"], 1, pad_len)
    batch["r_set_to_rep_atom"] = pad_dim(batch["r_set_to_rep_atom"], 1, pad_len)
    batch["disto_target"] = pad_dim(pad_dim(batch["disto_target"], 1, pad_len),
                                    2, pad_len)
    batch["frames_idx"] = pad_dim(batch["frames_idx"], 1, pad_len)
    batch["frame_resolved_mask"] = pad_dim(batch["frame_resolved_mask"], 1,
                                           pad_len)

    # Padding for token features
    batch["token_index"] = pad_dim(batch["token_index"], 1, pad_len)
    batch["residue_index"] = pad_dim(batch["residue_index"], 1, pad_len)
    batch["asym_id"] = pad_dim(batch["asym_id"], 1, pad_len)
    batch["entity_id"] = pad_dim(batch["entity_id"], 1, pad_len)
    batch["sym_id"] = pad_dim(batch["sym_id"], 1, pad_len)
    batch["mol_type"] = pad_dim(batch["mol_type"], 1, pad_len)
    batch["res_type"] = pad_dim(batch["res_type"], 1, pad_len)
    batch["disto_center"] = pad_dim(batch["disto_center"], 1, pad_len)
    batch["token_bonds"] = pad_dim(pad_dim(batch["token_bonds"], 1, pad_len), 2,
                                   pad_len)
    batch["token_pad_mask"] = pad_dim(batch["token_pad_mask"], 1, pad_len)
    batch["token_resolved_mask"] = pad_dim(batch["token_resolved_mask"], 1,
                                           pad_len)
    batch["token_disto_mask"] = pad_dim(batch["token_disto_mask"], 1, pad_len)
    batch["pocket_feature"] = pad_dim(batch["pocket_feature"], 1, pad_len)

    return batch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--structure_pairformer_engines_dir',
        type=Path,
        default=None,
        help=
        'The path to the directory containing the structure pairformer engines')
    parser.add_argument(
        '--confidence_pairformer_engines_dir',
        type=Path,
        default=None,
        help=
        'The path to the directory containing the confidence pairformer engines'
    )
    parser.add_argument('--original_torch',
                        action='store_true',
                        help='Use default torch')
    parser.add_argument('--torch_backend_config',
                        type=Path,
                        default=None,
                        help='The path to the torch backend config')
    parser.add_argument('--sample_dir',
                        type=Path,
                        default='sample',
                        help='The path to the directory containing the sample')
    parser.add_argument('--gpu_per_node',
                        type=int,
                        default=8,
                        help='The number of GPUs per node')
    parser.add_argument('--repeat',
                        type=int,
                        default=2,
                        help='The number of times to repeat the inference')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=100000,
                        help='The maximum sequence length to run')
    parser.add_argument('--min_seq_len',
                        type=int,
                        default=-1,
                        help='The minimum sequence length to run')
    parser.add_argument('--strategy',
                        type=str,
                        default="test",
                        help='The strategy to run')
    return parser.parse_args()


def run_single_rank(sample_dir: Path, model: nn.Module, rank: int,
                    dcp_size: int, device: torch.device,
                    predict_params: BoltzPredictionParams, strategy: str):
    # TODO: write docs for sample dir
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    torch.cuda.set_device(rank)

    pdb_ids = [
        ele.split('/feats_')[1][:4]
        for ele in glob.glob(f"{sample_dir.as_posix()}/feats*.pt")
    ]
    seqlens = []
    new_pdb_ids = []
    ids = json.load(open("sample/ids.json"))

    # for pdb_id in pdb_ids:
    for pdb_id, seqlen in ids.items():
        if seqlen <= args.max_seq_len and seqlen >= args.min_seq_len:
            seqlens.append(seqlen)
            new_pdb_ids.append(pdb_id)
            if rank == 0:
                logger.info(f"Load pdb_id: {pdb_id} with seqlen: {seqlen}")

    report_df = pd.DataFrame(columns=[
        "strategy", "pdb_id", "inference_time", "rmsd", "lddt", "seqlen"
    ])
    pdb_ids = new_pdb_ids
    v = sorted(zip(pdb_ids, seqlens), key=lambda x: x[1])
    pdb_ids = [x[0] for x in v]
    seqlens = [x[1] for x in v]
    for i, pdb_id in enumerate(pdb_ids):
        feats_path = f"{sample_dir.as_posix()}/feats_{pdb_id}.pt"
        batch = torch.load(feats_path, weights_only=False)
        for key, val in batch.items():
            if hasattr(val, "to"):
                batch[key] = val.to(device)
        pad_seqlen = (seqlens[i] + dcp_size - 1) // dcp_size * dcp_size
        batch = pad_batch(batch, pad_seqlen, seqlens[i])
        for l in range(args.repeat):
            torch.cuda.empty_cache()
            seed_everything(SEED)
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                output = model(
                    batch,
                    recycling_steps=predict_params.recycling_steps,
                    num_sampling_steps=predict_params.sampling_steps,
                    diffusion_samples=predict_params.diffusion_samples,
                    run_confidence_sequentially=True)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            if rank == 0 and l == args.repeat - 1:
                logger.info(
                    f"Sample {pdb_id} Sequence length: {seqlens[i]} - Loop {l} with inference time (GPU): {inference_time:.4f} seconds"
                )
                ref_output = torch.load(
                    f"{sample_dir.as_posix()}/pred_dict_{pdb_id}.pt",
                    weights_only=False)
                for key, val in ref_output.items():
                    if key == 'sample_atom_coords':
                        ref_output[key] = val.to(device)
                rmsd_value = kabsch_torch(
                    output['sample_atom_coords'].squeeze(0),
                    ref_output['sample_atom_coords'].squeeze(0))
                lddt_value = lddt(output['sample_atom_coords'],
                                  ref_output['sample_atom_coords'],
                                  batch['atom_resolved_mask'])
                logger.info(
                    f"Sample {pdb_id} Sequence length: {seqlens[i]} with RMSD: {rmsd_value:.4f} and LDDT: {lddt_value.mean().cpu().numpy():.4f}"
                )
                row = pd.DataFrame([{
                    "strategy":
                    strategy,
                    "pdb_id":
                    pdb_id,
                    "inference_time":
                    round(inference_time, 4),
                    "rmsd":
                    round(float(rmsd_value.cpu().numpy()), 4),
                    "lddt":
                    round(float(lddt_value.mean().cpu().numpy()), 4),
                    "seqlen":
                    seqlens[i],
                }])
                report_df = pd.concat([report_df, row], ignore_index=True)

    if rank == 0:
        report_df.to_csv(f"report_{strategy}.csv", index=False)


def main(args):
    import tensorrt_llm

    rank = tensorrt_llm.mpi_rank()
    world_size = tensorrt_llm.mpi_world_size()
    torch.cuda.set_device(rank % args.gpu_per_node)
    model, predict_params = create_original_model(device=torch.device("cuda"))
    logger.set_level("info")
    structure_pairformer = None
    confidence_pairformer = None
    dcp_size = 1
    context_address = None  # use only by trt-backend
    if args.structure_pairformer_engines_dir:
        with open(args.structure_pairformer_engines_dir / "config.json",
                  "r") as f:
            config = PairformerConfig.from_dict(
                json.load(f)["pretrained_config"])
            dcp_size = config.mapping.dcp_size
        structure_pairformer = PairformerBackendBuilder.build(
            config=config,
            checkpoint_dir=args.structure_pairformer_engines_dir,
            world_size=world_size,
            rank=rank,
            context_without_device_memory=True)
        setattr(model, "pairformer_module", structure_pairformer)
        context_address, _ = structure_pairformer.get_backend_workspace()
        assert args.confidence_pairformer_engines_dir is not None, "confidence pairformer engines dir is required"
        with open(args.confidence_pairformer_engines_dir / "config.json",
                  "r") as f:
            config = PairformerConfig.from_dict(
                json.load(f)["pretrained_config"])
        confidence_pairformer = PairformerBackendBuilder.build(
            config=config,
            checkpoint_dir=args.confidence_pairformer_engines_dir,
            world_size=world_size,
            rank=rank,
            context_without_device_memory=True,
            address=context_address)
        setattr(model.confidence_module, "pairformer_module",
                confidence_pairformer)
    else:
        # Using original torch or torch backend
        if args.original_torch:
            assert world_size == 1, "original torch only supports single GPU"
        else:
            with open(args.torch_backend_config) as f:
                config = PairformerConfig.from_dict(
                    json.load(f)["pretrained_config"])
            # Override mapping to ensure correct mapping for each rank
            config.mapping = Mapping(world_size=world_size,
                                     tp_size=config.mapping.tp_size,
                                     dcp_size=config.mapping.dcp_size,
                                     rank=rank)
            dcp_size = config.mapping.dcp_size
            structure_weights = convert_hf_pairformer(config, None, "structure")
            confidence_weights = convert_hf_pairformer(config, None,
                                                       "confidence")

            structure_pairformer = PairformerBackendBuilder.build(
                config=config,
                world_size=world_size,
                rank=rank,
                weights=structure_weights,
                load_weights_fn=torch_pairformer_load_fn)
            confidence_pairformer = PairformerBackendBuilder.build(
                config=config,
                world_size=world_size,
                rank=rank,
                weights=confidence_weights,
                load_weights_fn=torch_pairformer_load_fn)
            setattr(model, "pairformer_module", structure_pairformer)
            setattr(model.confidence_module, "pairformer_module",
                    confidence_pairformer)
    run_single_rank(sample_dir=args.sample_dir,
                    model=model,
                    rank=rank,
                    dcp_size=dcp_size,
                    device=torch.device("cuda"),
                    predict_params=predict_params,
                    strategy=args.strategy)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_arguments()
    main(args)
