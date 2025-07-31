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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import boltz.data.const as const
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from boltz.data.pad import pad_dim
from boltz.model.models.boltz2 import Boltz2
# isort: on
from pytorch_lightning import seed_everything
from score import kabsch_torch, lddt
from tensorrt_llm.logger import logger

from tensorrt_bionemo.hubs import load_hf_weights
from tensorrt_bionemo.models.boltz2 import Boltz2 as Boltz2Opt
from tensorrt_bionemo.models.boltz2 import (Boltz2AcceleratedModules,
                                            Boltz2Config)
from tensorrt_bionemo.models.helper import AcceleratedConfig
from tensorrt_bionemo.runtime import BackendType, SharedContextMemoryManager

SEED = 42
"""
NOTE:
    This script is used to run the demo of the Boltz2 model along with torch backbone from the original repo.
    It is used to verify the correctness of the TensorRT-BNM implementation. The inputs to model is dumped by `boltz predict`.
    For usage TRT-engines in production, please use _torch.backend for models.
"""


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzPredictionParams:
    recycling_steps: int = 3
    sampling_steps: int = 50
    diffusion_samples: int = 1
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False
    max_parallel_samples: Optional[int] = None


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


def create_original_model(
    checkpoint: str = None,
    device: torch.device = torch.device("cuda")
) -> nn.Module:
    if checkpoint is None:
        cached_file = load_hf_weights("boltz-2", return_raw=True)
    else:
        cached_file = checkpoint
    predict_params = BoltzPredictionParams()
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_module_args = MSAModuleArgs()
    steering_args = BoltzSteeringParams()

    # trainer = Trainer(accelerator="gpu", devices=1, precision="bf16-mixed")
    # with trainer.init_module():
    model: Boltz2 = Boltz2.load_from_checkpoint(
        cached_file,
        strict=True,
        predict_args=asdict(predict_params),
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_module_args=asdict(msa_module_args),
        steering_args=asdict(steering_args),
    )
    model.eval()

    return model, predict_params


def pad_batch(batch: dict, pad_seqlen: int, seqlen: int):
    # TODO: modify this function to pad the batch for the Boltz2
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
        '--structure_pairformer_ckpt',
        type=Path,
        default=None,
        help=
        'The path to the directory containing the structure pairformer engines')
    parser.add_argument(
        '--confidence_pairformer_ckpt',
        type=Path,
        default=None,
        help=
        'The path to the directory containing the confidence pairformer engines'
    )
    parser.add_argument(
        '--token_transformer_ckpt',
        type=Path,
        default=None,
        help='The path to the directory containing the token transformer engines'
    )
    parser.add_argument('--structure_pairformer_backend',
                        type=str,
                        default=BackendType.TORCH,
                        help='The backend to use for the structure pairformer')
    parser.add_argument('--confidence_pairformer_backend',
                        type=str,
                        default=BackendType.TORCH,
                        help='The backend to use for the confidence pairformer')
    parser.add_argument('--token_transformer_backend',
                        type=str,
                        default=BackendType.TORCH,
                        help='The backend to use for the token transformer')
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
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
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
            if hasattr(model.structure_module.score_model.token_transformer,
                       "reset"):
                """ TODO: Refactor for all backends to call reset method. """
                model.structure_module.score_model.token_transformer.reset()
            torch.cuda.empty_cache()
            seed_everything(SEED)
            np.random.default_rng(SEED)
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                # with torch.no_grad():
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
    tensorrt_llm.mpi_world_size()
    torch.cuda.set_device(rank % args.gpu_per_node)
    model, predict_params = create_original_model(device=torch.device("cuda"))
    logger.set_level("info")
    dcp_size = 1

    # Create optimized model with TensorRT backends
    manager = SharedContextMemoryManager()
    config = Boltz2Config.from_pretrained()

    config.structure_pairformer_config.set_dtype("bfloat16")
    config.token_transformer_config.set_dtype("bfloat16")
    config.confidence_pairformer_config.set_dtype("bfloat16")
    config.msa_module_config.set_dtype("bfloat16")

    acc_m = Boltz2AcceleratedModules(
        configs={
            "structure_pairformer":
            AcceleratedConfig(checkpoint=args.structure_pairformer_ckpt,
                              backend=args.structure_pairformer_backend,
                              default=config.structure_pairformer_config),
            "confidence_pairformer":
            AcceleratedConfig(checkpoint=args.confidence_pairformer_ckpt,
                              backend=args.confidence_pairformer_backend,
                              default=config.confidence_pairformer_config),
            "token_transformer":
            AcceleratedConfig(checkpoint=args.token_transformer_ckpt,
                              backend=args.token_transformer_backend,
                              default=config.token_transformer_config),
            "msa_module":
            AcceleratedConfig(checkpoint=None,
                              backend=BackendType.TORCH,
                              default=config.msa_module_config),
        })
    model = Boltz2Opt.optimize(model, acc_m, manager)

    run_single_rank(sample_dir=args.sample_dir,
                    model=model,
                    rank=rank,
                    dcp_size=dcp_size,
                    device=torch.device("cuda"),
                    predict_params=predict_params,
                    strategy=args.strategy)


if __name__ == "__main__":
    args = parse_arguments()
    # mp.set_start_method('spawn')
    main(args)
