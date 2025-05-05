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
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import boltz.data.const as const
import pandas as pd
import tensorrt as trt
import torch
import torch.nn as nn
from boltz.data.feature.pad import pad_dim
from boltz.model.model import Boltz1
# isort: on
from cuda import cudart
from pytorch_lightning import seed_everything
from score import kabsch_torch, lddt
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime import Session, TensorInfo
from tensorrt_llm.runtime.session import _scoped_stream

from tensorrt_bionemo.hf.checkpoints import load_hf_weights
from tensorrt_bionemo.mapping import Mapping

SEED = 42
"""
NOTE:
    This script is used to run the demo of the Boltz1 model along with torch backbone from the original repo.
    It is used to verify the correctness of the TensorRT-BNM implementation. The inputs to model is dumped by `botlz predict`.
    For usage TRT-engines in production, please use _torch.backend for models.
"""


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class PairformerTRT(nn.Module):
    """ TODO: Move this class to tensorrt_bionemo/runtime/modules and support CUDA graph"""

    def __init__(self,
                 engines_dir: Path,
                 world_size: int,
                 rank: int,
                 context_without_device_memory: bool = True,
                 address=None,
                 stream=None):
        super().__init__()
        self.engines_dir = engines_dir
        self.world_size = world_size
        self.runtime_rank = rank
        config_path = engines_dir / "config.json"
        with config_path.open("r") as f:
            self.config = json.load(f)
        # Sanity checks
        if 'pretrained_config' in self.config:  # new build api branch
            config_dtype = self.config['pretrained_config']['dtype']
            logger.info(f"Engine dtype: {config_dtype}")
            self.disable_custom_all_reduce = self.config['pretrained_config'][
                'disable_custom_all_reduce']
            self.tp_size = self.config['pretrained_config']['mapping'][
                'tp_size']
            self.dcp_size = self.config['pretrained_config']['mapping'][
                'dcp_size']
            assert world_size == self.world_size, \
                (f'Engine world size ({world_size}) != Runtime world size ({self.world_size})')
        self.engine_name = f"rank{self.runtime_rank}.engine"
        self.runtime_mapping = Mapping(world_size=self.world_size,
                                       rank=self.runtime_rank,
                                       tp_size=self.tp_size,
                                       dcp_size=self.dcp_size)
        if self.world_size > 1 and not self.disable_custom_all_reduce:
            # init_all_reduce_helper()
            _, self.workspace = CustomAllReduceHelper.allocate_workspace(
                self.runtime_mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.runtime_mapping.tp_size))
        self.serialize_path = os.path.join(self.engines_dir, self.engine_name)
        with open(self.serialize_path, 'rb') as f:
            engine_buffer = f.read()
            assert engine_buffer is not None
        logger.info(f"Deserialize engine from {self.serialize_path}")
        # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        self.runtime = trt.Runtime(logger.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        self.device_memory_size = self.engine.device_memory_size_v2
        self.address = None
        if not context_without_device_memory:
            self.context = self.engine.create_execution_context()
            with _scoped_stream() as stream:
                self.context.set_optimization_profile_async(0, stream)
        else:
            self.context = self.engine.create_execution_context_without_device_memory(
            )
            if address is None:
                address = CUASSERT(cudart.cudaMalloc(
                    self.device_memory_size))[0]
            self.context.set_device_memory(address, self.device_memory_size)
            self.address = address
            with _scoped_stream() as stream:
                self.context.set_optimization_profile_async(0, stream)
        # Initialize session
        self.session = Session()
        self.session._runtime = self.runtime
        self.session._context = self.context
        self.session.engine = self.engine

        self.session._print_engine_info()
        self.engine = self.session.engine
        logger.info(
            f"The memory required by the largest profile: {self.engine.device_memory_size_v2}"
        )
        self.context = self.session.context

        self.op_profile_map = {}
        num_optimization_profiles = self.engine.num_optimization_profiles
        for i in range(num_optimization_profiles):
            mask_dims = self.engine.get_tensor_profile_shape("mask", i)
            min_opt = mask_dims[0]
            max_opt = mask_dims[-1]
            min_s = min_opt[0]
            max_s = max_opt[0]
            self.op_profile_map[(min_s, max_s)] = i
        self.curr_profile = 0
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.current_stream().cuda_stream

    def switch_opt_profile(self, input_length: int):
        found_profile = -1
        for k, v in self.op_profile_map.items():
            if k[0] <= input_length <= k[1]:
                found_profile = v
        if found_profile == -1:
            raise ValueError(
                f"No suitable optimization profile found for the current rank. "
                f"Please check the engine configuration.")
        if found_profile != self.curr_profile:
            self.curr_profile = found_profile
            with _scoped_stream() as stream:
                self.context.set_optimization_profile_async(
                    self.curr_profile, stream)

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure the inputs are contiguous
        if not s.is_contiguous():
            s = s.contiguous()
        if not z.is_contiguous():
            z = z.contiguous()
        if not mask.is_contiguous():
            mask = mask.contiguous()
        if not pair_mask.is_contiguous():
            pair_mask = pair_mask.contiguous()
        if s.ndim == 3:
            s = s.squeeze(0)
            z = z.squeeze(0)
            mask = mask.squeeze(0)
            pair_mask = pair_mask.squeeze(0)

        inputs = {"s": s, "z": z, "mask": mask, "pair_mask": pair_mask}
        self.switch_opt_profile(s.shape[0])
        output_info = self.session.infer_shapes([
            TensorInfo("s", dtype=trt.DataType.FLOAT, shape=s.shape),
            TensorInfo("z", dtype=trt.DataType.FLOAT, shape=z.shape),
            TensorInfo("mask", dtype=trt.DataType.FLOAT, shape=mask.shape),
            TensorInfo(
                "pair_mask", dtype=trt.DataType.FLOAT, shape=pair_mask.shape),
        ], self.context)
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in output_info
        }
        if self.world_size > 1 and not self.disable_custom_all_reduce:
            inputs["all_reduce_workspace"] = self.workspace
        ok = self.session.run(inputs,
                              outputs,
                              self.stream,
                              context=self.context)
        assert ok, "Runtime execution failed"
        s = outputs["output_s"].unsqueeze(0)
        z = outputs["output_z"].unsqueeze(0)
        return s, z


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


class MSAModuleTorch(nn.Module):

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self._original_module = original_module

        for layer in self._original_module.layers:
            layer.tri_att_start.layer_norm = ForceFP32(
                layer.tri_att_start.layer_norm)
            layer.tri_att_start.linear = ForceFP32(layer.tri_att_start.linear)
            layer.tri_att_start.mha.linear_q = ForceFP32(
                layer.tri_att_start.mha.linear_q)
            layer.tri_att_start.mha.linear_k = ForceFP32(
                layer.tri_att_start.mha.linear_k)
            layer.tri_att_start.mha.linear_v = ForceFP32(
                layer.tri_att_start.mha.linear_v)

            layer.tri_att_end.layer_norm = ForceFP32(
                layer.tri_att_end.layer_norm)
            layer.tri_att_end.linear = ForceFP32(layer.tri_att_end.linear)
            layer.tri_att_end.mha.linear_q = ForceFP32(
                layer.tri_att_end.mha.linear_q)
            layer.tri_att_end.mha.linear_k = ForceFP32(
                layer.tri_att_end.mha.linear_k)
            layer.tri_att_end.mha.linear_v = ForceFP32(
                layer.tri_att_end.mha.linear_v)

    def forward(self, *args, **kwargs):
        return self._original_module(*args, **kwargs)


class PairformerTorch(nn.Module):

    def __init__(self, original_module: nn.Module, name="structure"):
        super().__init__()
        self._original_module = original_module
        self._name = name

    def forward(self, s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor, **kwargs):
        s, z = self._original_module(s, z, mask, pair_mask, **kwargs)
        return s, z


class FP32StructureModule(nn.Module):

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self._original_module = original_module

    def sample(self, *args, **kwargs):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        output = self._original_module.sample(*args, **kwargs)
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
            with torch.inference_mode():
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
    if args.structure_pairformer_engines_dir:
        structure_pairformer = PairformerTRT(
            args.structure_pairformer_engines_dir,
            world_size,
            rank,
            context_without_device_memory=True)
        setattr(model, "pairformer_module", structure_pairformer)
    else:
        setattr(model, "pairformer_module",
                PairformerTorch(model.pairformer_module, name="structure"))
    if args.confidence_pairformer_engines_dir:
        address = None
        if isinstance(structure_pairformer,
                      PairformerTRT):  # sharing same device memory
            address = structure_pairformer.address
        confidence_pairformer = PairformerTRT(
            args.confidence_pairformer_engines_dir,
            world_size,
            rank,
            context_without_device_memory=True,
            address=address)
        setattr(model.confidence_module, "pairformer_module",
                confidence_pairformer)
    else:
        setattr(
            model.confidence_module, "pairformer_module",
            PairformerTorch(model.confidence_module.pairformer_module,
                            name="confidence"))
    setattr(model, "structure_module",
            FP32StructureModule(model.structure_module))

    dcp_size = 1
    if structure_pairformer:
        dcp_size = structure_pairformer.runtime_mapping.dcp_size
    elif confidence_pairformer:
        dcp_size = confidence_pairformer.runtime_mapping.dcp_size
    run_single_rank(sample_dir=args.sample_dir,
                    model=model,
                    rank=rank,
                    dcp_size=dcp_size,
                    device=torch.device("cuda"),
                    predict_params=predict_params,
                    strategy=args.strategy)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
